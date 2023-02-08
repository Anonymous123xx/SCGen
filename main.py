from utils import *
import sys
import pickle
from nltk.translate.meteor_score import single_meteor_score



class Config:
    def __init__(self):
        self.BATCH_SIZE = 32
        self.EPOCH = 120
        self.MAX_INPUT_SIZE = 400
        self.MAX_OUTPUT_SIZE = 20
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.MODEL_SIZE = 512
        self.DICT_SIZE_1 = 50010
        self.DICT_SIZE_2 = 1010
        self.NUM_LAYER = 1
        self.DROPOUT = 0.25
        self.LR = 1e-4




def func(mm):
    ast = mm['ast']
    k = 0
    mask = [True for _ in ast]
    for i in range(len(ast)):
        if ast[k]['num']==ast[i]['num']: k = i
        if ast[i]['num']==0: mask[i] = False
    return get_sbt(k,ast,mask)

def func_mix(mm):
    sbt = func(mm)
    code = nltk.word_tokenize(mm['code'].strip())
    code = [split_word(x) for x in code]
    code = [x for word in code for x in word]
    return (sbt,code)
    
def func_relate(mm, expand_order=1):    #find expanded codes and LCA
    ast = mm['ast']
    code = nltk.word_tokenize(mm['code'].strip())
    code = [split_word(x) for x in code]
    code = [x for word in code for x in word]
    k = 0
    mask = [True for _ in ast]
    for i in range(len(ast)):
        if ast[k]['num'] == ast[i]['num']: k = i  
        if ast[i]['num'] == 0: mask[i] = False  
    mask_deepcom = mask     
    for i in range(k):
        mask_deepcom[i] = False
    inline_num = 0
    root_deepcom = k
    for i in mask[root_deepcom:]:
        if i:
            inline_num += 1  
    if inline_num >= 100:  
        relate = get_sbt(root_deepcom, ast, mask)
        for i in range(root_deepcom):
            mask[i] = False
        # return mask
        return (relate, code)

    li = traverse(k, mm['ast'], mask)
    mask = [False for _ in ast]
    for x in li: mask[x] = True
    for _ in range(expand_order):
        names = find_name(ast, mask)
        scopes = []
        for name in names:
            scope = find_name_scope(ast, name)
            colors = []
            for i in names[name]: colors.append(scope[i])
            for i in range(len(scope)):
                if scope[i] in colors:
                    scope[i] = True
                else:
                    scope[i] = False
            scopes.append((name, scope))
        mask_ = find_scope(ast, scopes)
        mask = [x or y for x, y in zip(mask, mask_)]

    num = [0 for _ in ast]
    for i in range(len(ast) - 1, -1, -1):
        node = ast[i]
        num[i] = 1 if mask[i] else 0
        for child in node['children']:
            num[i] += num[child]
    for node in ast:
        i = node['id']
        if num[i] > 0:
            mask[i] = True
            if node['type'] == 'ForStatement' or node['type'] == 'WhileStatement' or node['type'] == 'IfStatement':
                child = node['children'][0]
                li = traverse(child, ast)
                for elem in li: mask[elem] = True
    num = [0 for _ in ast]
    for i in range(len(ast) - 1, -1, -1):
        node = ast[i]
        num[i] = 1 if mask[i] else 0
        for child in node['children']:
            num[i] += num[child]    

    root_relate = 0
    for i in range(len(ast)):
        if num[i] == num[0]:
            root_relate = i     
    if num[root_relate] <= 100:     
        relate = get_sbt(root_relate, ast, mask)
        for i in range(root_relate):
            mask[i] = False
        # return mask
        return (relate, code)
    else:
        root_relate = root_deepcom  
        while num[root_relate] <= 100:     
            root_relate = ast[root_relate]['parent']
        exceed_node_num =  num[root_relate] - 100
        reserved_node = []  
        for index,i in enumerate(mask_deepcom):     
            if i:
                reserved_node.append(index)
        #
        for token in mm['ast']:
            if token['parent'] == -1:
                token['depth'] = 0
            else:
                token['depth'] = mm['ast'][token['parent']]['depth'] + 1  
        distance_list = []  
        for index,i in enumerate(mask):
            if index < root_relate:
                mask[index] = False
            else:
                if i and (index not in reserved_node):  
                    path_distance = []  
                    for j in reserved_node:
                        start = mm['ast'][index]   
                        end = mm['ast'][j]
                        path1,path2 =[],[]
                        path1.append(start)
                        path2.append(end)
                        while start != end:
                            if start['depth'] >= end['depth']:
                                start = mm['ast'][start['parent']]
                                path1.append(start)
                            else:
                                end = mm['ast'][end['parent']]
                                path2 = [end] + path2
                        path = path1 + path2[1:]
                        path_distance.append(len(path))
                    total_distance = sum(path_distance)     
                    distance_list.append(total_distance)
                    mask[index] = total_distance    
        distance_list.sort(reverse=True)    
        threshold = distance_list[exceed_node_num + 1:]     
        for i in range(len(mask)):
            if (mask[i] != True) and (mask[i] != False) and (mask[i] in threshold):
                mask[i] = True
            if (mask[i] != True) and (mask[i] != False) and (mask[i] not in threshold):
                mask[i] = False
        relate = get_sbt(root_relate, ast, mask)
        # return mask
        return (relate, code)
    
    
    

def get_batch(path,f,config,in_w2i,out_w2i):
    batch_in1, batch_in2, batch_out = [], [], []
    in1_w2i, in2_w2i = in_w2i
    for tu in get_one(f,path):
       in_, out = tu
       in1, in2 = in_
       in1 = [in1_w2i[x] if x in in1_w2i else in1_w2i['<UNK>'] for x in in1]
       in2 = [in2_w2i[x] if x in in2_w2i else in2_w2i['<UNK>'] for x in in2]
       out = [out_w2i[x] if x in out_w2i else out_w2i['<UNK>'] for x in out]
       in1 = in1[:config.MAX_INPUT_SIZE]
       in2 = in2[:config.MAX_INPUT_SIZE]
       out = out[:config.MAX_OUTPUT_SIZE]
       if len(in1)==0: continue
       if len(in2)==0: continue
       if len(out)==0: continue
       batch_in1.append(in1)
       batch_in2.append(in2)
       batch_out.append(out)
       if len(batch_out)>=config.BATCH_SIZE:
           yield (batch_in1, batch_in2), batch_out
           batch_in1, batch_in2, batch_out = [], [], []
    if len(batch_out)>0:
        yield (batch_in1, batch_in2), batch_out




def get_batch_mix(path,f,config,in_w2i,out_w2i):
    batch_in1, batch_in2, batch_in3, batch_out = [], [], [], []
    in_w2i, in3_w2i = in_w2i
    in1_w2i, in2_w2i = in_w2i
    for tu in get_one(f,path):
       in_, out = tu    #in_ is ((（type,value),code),nl)
       in_, in3 = in_
       in1,in2 = in_
       in1 = [in1_w2i[x] if x in in1_w2i else in1_w2i['<UNK>'] for x in in1]
       in2 = [in2_w2i[x] if x in in2_w2i else in2_w2i['<UNK>'] for x in in2]
       in3 = [in3_w2i[x] if x in in3_w2i else in3_w2i['<UNK>'] for x in in3]
       out = [out_w2i[x] if x in out_w2i else out_w2i['<UNK>'] for x in out]
       in1 = in1[:config.MAX_INPUT_SIZE]
       in2 = in2[:config.MAX_INPUT_SIZE]
       in3 = in3[:config.MAX_INPUT_SIZE]
       out = out[:config.MAX_OUTPUT_SIZE]
       if len(in1)==0: continue
       if len(in2)==0: continue
       if len(in3)==0: continue
       if len(out)==0: continue
       batch_in1.append(in1)
       batch_in2.append(in2)
       batch_in3.append(in3)
       batch_out.append(out)
       if len(batch_out)>=config.BATCH_SIZE:
           yield ((batch_in1,batch_in2),batch_in3), batch_out
           batch_in1, batch_in2, batch_in3, batch_out = [], [], [], []
    if len(batch_out)>0:
        yield ((batch_in1,batch_in2),batch_in3), batch_out


def get_batch_relate(path,f,config,in_w2i,out_w2i):
    return get_batch_mix(path,f,config,in_w2i,out_w2i)


if __name__ == "__main__":
    model_name = sys.argv[1]
    path = sys.argv[2]
    start = -1
    config = Config()
    logger = get_logger('{}/{}_logging.txt'.format(path,model_name))
    type_w2i = pickle.load(open('{}/type_.w2i'.format(path),'rb'))
    value_w2i = pickle.load(open('{}/value_.w2i'.format(path),'rb'))
    code_w2i = pickle.load(open('{}/code_.w2i'.format(path),'rb'))
    api_w2i = pickle.load(open('{}/api_.w2i'.format(path), 'rb'))
    nl_w2i = pickle.load(open('{}/nl_.w2i'.format(path),'rb'))
    type_i2w = pickle.load(open('{}/type_.i2w'.format(path),'rb'))
    value_i2w = pickle.load(open('{}/value_.i2w'.format(path),'rb'))
    code_i2w = pickle.load(open('{}/code_.i2w'.format(path),'rb'))
    api_i2w = pickle.load(open('{}/api_.i2w'.format(path), 'rb'))
    nl_i2w = pickle.load(open('{}/nl_.i2w'.format(path),'rb'))


    if model_name == 'BCGen':
        from mix import *
        in_w2i = ((type_w2i,value_w2i),code_w2i)
        get_batch = get_batch_relate
        f = func_relate
        encoder1 = SBTEncoder(config)
        encoder2 = NormalEncoder(config)
        model = Model(config,encoder1,encoder2)
        if start != -1:
            model.load('{}/model/{}_{}'.format(path, model_name, start), model_name)
    out_w2i = nl_w2i
    best_bleu = 0.
    for epoch in range(start + 1, config.EPOCH):
        loss = 0.
        for step,batch in enumerate(get_batch('{}/train.json'.format(path),f,config,in_w2i,out_w2i)):
            batch_in, batch_out = batch
            loss += model(batch_in,True,batch_out)
            logger.info('Epoch: {}, Batch: {}, Loss: {}'.format(epoch,step,loss/(step+1)))

        preds = []
        refs = []
        bleu1 = bleu2 = bleu3 = bleu4 = meteor = rouge = 0.
        len_rouge_preds = 0
        for step,batch in enumerate(get_batch('{}/valid.json'.format(path),f,config,in_w2i,out_w2i)):
            batch_in, batch_out = batch
            pred = model(batch_in,False)
            preds += pred
            refs += batch_out
            len_rouge_preds += len(pred)
            for x,y in zip(batch_out,pred):
                bleu1 += calc_bleu([x],[y],1)
                bleu2 += calc_bleu([x], [y], 2)
                bleu3 += calc_bleu([x], [y], 3)
                bleu4 += calc_bleu([x], [y], 4)
                meteor += single_meteor_score(' '.join([str(z) for z in x]),' '.join([str(z) for z in y]))
                if len(y) > 0:
                    rouge += myrouge([x],y)
                else:
                    len_rouge_preds -= 1
            logger.info('Epoch: {}, Batch: {}, BLEU-1: {}, BLEU-2: {},BLEU-3: {}, BLEU-4: {}, METEOR: {}, ROUGE-L: {}'.format(epoch,step,bleu1/len(preds),bleu2/len(preds),bleu3/len(preds),bleu4/len(preds),meteor/len(preds),rouge/len_rouge_preds))
        logger.info('Epoch: {}, Batch: {}, BLEU-1: {}, BLEU-2: {},BLEU-3: {}, BLEU-4: {}, METEOR: {}, ROUGE-L: {}'.format(epoch, step, bleu1/len(preds), bleu2/len(preds), bleu3/len(preds), bleu4/len(preds),meteor/len(preds), rouge/len_rouge_preds))

        if bleu4>best_bleu:
            best_bleu = bleu4
            model.save('{}/model/{}_{}'.format(path,model_name,epoch),model_name)

        preds = []
        refs = []
        bleu1 = bleu2 = bleu3 = bleu4 = meteor = rouge = 0.
        len_rouge_preds = 0
        for step,batch in enumerate(get_batch('{}/test.json'.format(path),f,config,in_w2i,out_w2i)):
            batch_in, batch_out = batch
            pred = model(batch_in,False)
            preds += pred
            refs += batch_out
            len_rouge_preds += len(pred)
            for x,y in zip(batch_out,pred):
                bleu1 += calc_bleu([x], [y], 1)
                bleu2 += calc_bleu([x], [y], 2)
                bleu3 += calc_bleu([x], [y], 3)
                bleu4 += calc_bleu([x], [y], 4)
                meteor += single_meteor_score(' '.join([str(z) for z in x]),' '.join([str(z) for z in y]))
                if len(y) > 0:
                    rouge += myrouge([x],y)
                else:
                    len_rouge_preds -= 1
            logger.info('Epoch: {}, testBatch: {}, BLEU-1: {}, BLEU-2: {},BLEU-3: {}, BLEU-4: {}, METEOR: {}, ROUGE-L: {}'.format(epoch, step, bleu1 / len(preds), bleu2 / len(preds), bleu3 / len(preds), bleu4 / len(preds), meteor / len(preds), rouge / len_rouge_preds))
        logger.info('Epoch: {}, testBatch: {}, BLEU-1: {}, BLEU-2: {},BLEU-3: {}, BLEU-4: {}, METEOR: {}, ROUGE-L: {}'.format(epoch, step, bleu1 / len(preds), bleu2 / len(preds), bleu3 / len(preds), bleu4 / len(preds), meteor / len(preds), rouge / len_rouge_preds))

    



