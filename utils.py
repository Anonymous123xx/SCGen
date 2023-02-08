import regex as re
import nltk
from nltk.translate.bleu_score import *
from nltk.stem import WordNetLemmatizer
import logging, json
from collections import defaultdict

def split_word(x):
    xs = x.split('.')
    li = []
    for x in xs:
        _li = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', x)).split()
        for elem in _li:
            li += elem.split('_')
        li.append('.')
    li = li[:-1]
    li = [x.lower() for x in li]
    return li

def normalize_word(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    res = []
    for word, pos in pos_tags:
        if pos.startswith('J'):
            tmp = lemmatizer.lemmatize(word, 'a')
        elif pos.startswith('V'):
            tmp = lemmatizer.lemmatize(word, 'v')
        elif pos.startswith('N'):
            tmp = lemmatizer.lemmatize(word, 'n')
        elif pos.startswith('R'):
            tmp = lemmatizer.lemmatize(word, 'r')
        else:
            tmp = lemmatizer.lemmatize(word)
        res.append(tmp)
    return res

def get_type(node):
    return [node['type']]

def get_value(node):
    li = split_word(node['value'])
    if len(li)==0: li = ['']
    return li

def get_node(node):
    t = get_type(node)
    v = get_value(node)
    t = [t[0] for _ in range(len(v))]
    return (t,v)


def traverse(root,ast,mask=None):
    li = []
    if (not mask) or (mask and mask[root]): li += [root]
    for child in ast[root]['children']:
        li += traverse(child,ast,mask)        
    return li

def get_sbt(root,ast,mask=None):
    node = get_node(ast[root])
    li_t, li_v = [], []
    if (not mask) or (mask and mask[root]):
        li_t += ['('] + node[0]
        li_v += ['('] + node[1]
    for child in ast[root]['children']:
        li = get_sbt(child,ast,mask)
        li_t += li[0]
        li_v += li[1]
    if (not mask) or (mask and mask[root]):
        li_t += [')'] + node[0]
        li_v += [')'] + node[1]
    return li_t, li_v

def calc_bleu(refs,preds,order=4):
    total_score = 0.0
    count = 0
    cc = SmoothingFunction()
    tu = [1./order for _ in range(order)]
    tu = tuple(tu)
    for ref,pred in zip(refs,preds):
        score = nltk.translate.bleu([ref],pred,tu,smoothing_function=cc.method2)
        total_score += score
        count += 1
    avg_score = total_score / count
    return avg_score

def myrouge(refs,pred,beta=1e2,eps=1e-2):
    max_score = 0.
    for ref in refs:
        dp = []
        for i in range(len(ref)):
            dp.append([])
            for j in range(len(pred)):
                dp[i].append(0)
                if i==0 or j==0:
                    if ref[i]==pred[j]:     
                        dp[i][j] = 1
                    if i>0: dp[i][j] = max([dp[i][j],dp[i-1][j]])
                    elif j>0: dp[i][j] = max([dp[i][j],dp[i][j-1]])
        for i in range(1,len(ref)):
            for j in range(1,len(pred)):
                dp[i][j] = max([dp[i][j-1],dp[i-1][j]])     
                if pred[j]==ref[i]:
                    dp[i][j] = max([dp[i][j],dp[i-1][j-1]+1])   
        lcs = max([eps,1.*dp[len(ref)-1][len(pred)-1]])
        rec = lcs / len(ref)        
        pre = lcs / len(pred)
        score = (1. + beta**2) * rec * pre / (rec + beta**2 * pre)
        max_score = max([score,max_score])
    return max_score

def get_logger(path):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(path)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def word2index(sentence,mdict):
    return [mdict[x] if x in mdict else mdict['<UNK>'] for x in sentence]

def index2word(sentence,mdict):
    return [mdict[x] for x in sentence]

def get_one(func,path):
    with open(path) as f:
        for line in f:
            mm = json.loads(line)
            if len(mm['ast'])==0: continue
            if mm['ast'][0]['num']==0: continue
            if len(mm['nl'].strip())==0: continue
            if len(mm['code'].strip())==0: continue
            in_ = func(mm)
            out = nltk.word_tokenize(mm['nl'].strip())
            out = [split_word(x) for x in out]
            out = [elem for x in out for elem in x]
            out = ' '.join(out)
            out = normalize_word(out)
            yield in_, out

def find_name(ast,mask):
    names = defaultdict(list)
    for node in ast:
        parent = ast[node['parent']]
        if mask[node['id']]:
            if node['type'] == 'MemberReference':
                names[node['value']].append(node['id'])
            elif node['type'] == 'FormalParameter':
                names[node['value']].append(node['id'])
            elif node['type'] == 'VariableDeclarator':
                names[node['value']].append(node['id'])
            elif node['type'] == 'MethodInvocation':
                names[node['value'].split('.')[0]].append(node['id'])
    return names

def find_scope(ast,names):
    mask = [False for _ in ast]
    for name,scope in names:
        for node in ast:
            if not scope[node['id']]: continue
            if mask[node['id']]: continue
            if node['value']==name or node['value'].split('.')[0]==name:
                x = node
                while True:
                    if x['type']=='MethodDeclaration': break
                    if x['type'].find('Statement')!=-1: break
                    if x['type']=='LocalVariableDeclaration': break
                    if x['type']=='FormalParameter': break
                    x = ast[x['parent']]
                if x['type']!='MethodDeclaration':
                    if x['type']!='ForStatement' and x['type']!='WhileStatement' and x['type']!='IfStatement':
                        li = traverse(x['id'],ast)
                        for elem in li: mask[elem] = True
                    else:
                        li = traverse(x['children'][0],ast)
                        for elem in li: mask[elem] = True
    return mask

def find_name_scope(ast,name):
    scope = [0 for _ in ast]
    def find_name_scope_(root,color,max_color):
        for child in ast[root]['children']:
            if ast[root]['type']=='LocalVariableDeclaration' and ast[child]['type']=='VariableDeclarator' and ast[child]['value']==name:
                max_color += 1
                color = max_color
        scope[root] = color
        color_ = color
        for child in ast[root]['children']:
            color_,max_color_ = find_name_scope_(child,color_,max_color)
            max_color = max([max_color,max_color_])
        return color,max_color
    _ = find_name_scope_(0,0,0)
    return scope
