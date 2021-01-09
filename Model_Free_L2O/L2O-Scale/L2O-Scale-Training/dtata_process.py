# this file is made to process the valid.txt and train.txt in custom folder so that to train yolo of our own dataset
import collections
import pdb
def process(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        src = f.read().strip()
    # src=src.replace(' ','')
    # src=src.replace('\n','')
    src = src.split('\n')
    # pdb.set_trace()
    target = []
    for i, item in enumerate(src):
        item = item.lstrip()
        # print(item.split(' '))
        feature = item.split(':')
        # print(feature)
        if feature[0] == 'timeline_label':
            if "MB" in feature[1]:
                target.append(feature[1][1:])
        # if feature[0][:-1] == 'node_name':
        #     target.append(feature[1])
            # print(revised)
    #     elif feature[0][:-1] == 'requested_bytes':
    #         device = src[i + 1].lstrip().split(' ')
    #         if device[0][:-1] == 'allocator_name':
    #             target.append(device[0] + device[1])
    #         device1 = src[i + 2].lstrip().split(' ')
    #         # print(device1)
    #         if device1[0][:-1] == 'allocator_name':
    #             target.append(device1[0] + device1[1])
    #
    #         if len(feature[1]) <= 3:
    #             target.append(feature[0] + feature[1] + ' bytes')
    #         elif 3 < len(feature[1]) < 6:
    #             target.append(feature[0] + str(int(feature[1])/1024) + ' KB')
    #         elif 6 < len(feature[1]) < 9:
    #             target.append(feature[0] + str(int(feature[1])/1024/1024) + ' MB')
    #     elif feature[0][:-1] == 'allocated_bytes':
    #         # device1 = src[i + 1].lstrip().split(' ')
    #         # print(device1)
    #         # if device1[0][:-1] == 'allocator_name':
    #         #     target.append(device1[0] + device1[1])
    #         if len(feature[1]) <= 3:
    #             target.append(feature[0] + feature[1] + ' bytes')
    #         elif 3 < len(feature[1]) < 6:
    #             target.append(feature[0] + str(int(feature[1])/1024) + ' KB')
    #         elif 6 < len(feature[1]) < 9:
    #             target.append(feature[0] + str(int(feature[1])/1024/1024) + ' MB')
    #
    # revised = []
    # for si, string in enumerate(target):
    #     if si < len(target) - 2:
    #         if string[0] == '"' and target[si+1][0] == '"':
    #             continue
    #         else:
    #             revised.append(string)
    print(target)
    print(len(target))
    # print(revised)
    # print(len(revised))
    # src_vocab=[]
    # [src_vocab.append(x) for x in src if x not in src_vocab and len(src_vocab)<5000]
    # random.shuffle(src_vocab)
    with open('report.txt','w',encoding='utf-8') as f2:
        for x in target:
            f2.write(str(x)+'\n')
        
def post_process1(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        src = f.read().strip()
    # src=src.replace(' ','')
    # src=src.replace('\n','')
    src = src.split('\n')
    # pdb.set_trace()
    target = []
    rnn_calc = 0
    for i, item in enumerate(src):
        item = item.lstrip()
        x1 = item.find('[')
        x2 = item.rfind(']')
        # print(x1, x2)
        sub_seq = item[x1 + 1: x2 - 1]
        # print(sub_seq)
        feature = sub_seq.split(' ')
        # print(feature)
        # num = float(feature[1].strip('MB'))
        # print(num)
        # rnn_calc += num
        # print(rnn_calc)
        # if num < 100:
        #     rnn_calc += num
        target.append(feature[1].strip('MB'))
    print(rnn_calc)
    with open('fine-grained-report.txt','w',encoding='utf-8') as f3:
        for x in target:
            f3.write(str(x)+'\n')
        # f3.write('gpu-usage-rnn--1-optimizer-iteration; ' + str(rnn_calc) + '\n')
            
def analysis(objects):
    conv = 0
    lstm = 0
    for item in objects:
        if len(item) < 4:
            lstm += float(item)
        else:
            conv += float(item)
    return conv, lstm
    
def post_process2(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        src = f.readlines()
    # src=src.replace(' ','')
    # src=src.replace('\n','')
    # src = src.split('\n')
    # pdb.set_trace()
    result = collections.defaultdict()
    sub_part = []
    for i, item in enumerate(src):
        # print(item)
        if item == '\n':
            conv, lstm = analysis(sub_part)
            result['iteration{}'.format(i)] = [conv, lstm]
            # result['lstm{}'.format(i)] = lstm
            sub_part = []
        else:
            sub_part.append(item.strip('\n'))
        
        
        
    with open('revised-report.txt','w',encoding='utf-8') as f3:
        print(len(result))
        
        for ki, kv in enumerate(result.values()):
            print(kv)
            f3.write('optimizee iteration: {}'.format(ki) + '\n')
            f3.write('conv-gpu-usage' + ':' + str(round(kv[0], 1)) + 'MB' + '\n')
            f3.write('lstm-gpu-usage' + ':' + str(round(kv[1], 1)) + 'MB' + '\n')
            f3.write('\n')


def post_process3(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        src = f.readlines()
    # src=src.replace(' ','')
    # src=src.replace('\n','')
    # src = src.split('\n')
    # pdb.set_trace()
    result = collections.defaultdict()
    grad_calc = 0
    cpnv_calc = 0
    for i, item in enumerate(src):
        # print(item)
        value = float(item.strip('\n'))
        if value < 100:
            grad_calc += value
    # grad_calc += 239.3
    print(grad_calc)

def post_process4(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        src = f.readlines()
    # src=src.replace(' ','')
    # src=src.replace('\n','')
    # src = src.split('\n')
    # pdb.set_trace()
    # result = collections.defaultdict()
    tail = 0
    last = src[1725:]
    for i, item in enumerate(last):
        tail += float(item.strip('\n'))
    print(tail)
    src = src[1362:]
    record = 0
    result = []
    conv = []
    grad_calc = 0
    # cpnv_calc = 0
    for i, item in enumerate(src):
        # print(item)
        value = float(item.strip('\n'))
        if value < 100:
            record += value
            if i == (len(src) - 1):
                result.append(record)
                
        elif value > 400:
            record += value
            result.append(record)
            record = 0
    # grad_calc += 239.3
    # print(grad_calc)
    with open('grad-report.txt', 'w', encoding='utf-8') as f3:
        f3.write('total graident memory usage: ' + str(sum(result)) + 'MB' + '\n')

        for ki, kv in enumerate(result):
            print(kv)
            f3.write('optimizee iteration-{}: '.format(ki + 1) + str(round(kv, 1)) + 'MB' + '\n')
        # f3.write('\n')
        # for ki, kv in enumerate(conv):
        #     print(kv)
        #     f3.write(str(kv) + '\n')
   
# filenames = ['report.txt']
# for filename in filenames:
#     post_process1(filename)

# filenames2 = ['run.txt']
# for filename in filenames2:
#     process(filename)


# filenames = ['fine-grained-report.txt']
# for filename in filenames:
#     post_process3(filename)
    
filenames = ['fine-grained-report.txt']
for filename in filenames:
    post_process4(filename)