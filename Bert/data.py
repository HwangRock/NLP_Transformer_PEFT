from random import random, randrange, randint, shuffle, choice


# mask 생성 함수, mask를 하는 이유는 MLM 목표함수때문이다.
def create_mask(tokens, mask_cnt, vocab):

    #  subword를 붙여서 단어를 만드는 과정.
    card_idx = []  # 각 단어의 인덱스를 연결된 서브워드끼리 짝지어서 리스트에 넣음.
    for (i, token) in enumerate(tokens):  # 인덱스하고 token을 같이 순회
        if token == "CLS" or token == "SEP":  # 단어 토큰이 아니라면 건너뛴다.
            continue

        if len(card_idx) > 0 and not token.startswith(u"\u2581"):  # u2581은 _다. sentencepiece에서 서브워드를 구분하기 위함.
            card_idx[-1].append(i)  # 연결된 서브워드이므로 마지막 단어와 연결되야하므로 마지막 리스트에 넣는다.
        else:
            card_idx.append([i])  # 새로운 단어이므로 새로운 리스트를 만들고, 시작은 해당 인덱스로 하게 만듦.

    shuffle(card_idx)  # overfitting을 막기위해서 셔플.

    # mask할 토큰을 결정하는 과정.
    mask_lms = []
    for index_set in card_idx:
        if len(mask_lms) >= mask_cnt:  # mask의 수가 데이터의 15%를 넘으면 중단
            break
        if len(mask_lms) + len(index_set) > mask_cnt:
            continue
        for index in index_set:
            if random() <= 0.8:
                state = "mask_token"  # 80%의 확률로 토큰이 mask가 됨.
            else:
                if random() < 0.5:
                    state = tokens[index]  # 10%의 확률로 토큰이 그대로가 됨.
                else:
                    state = choice(vocab)  # 10퍼센트의 확률로 토큰이 다른 단어로 바뀜.
            mask_lms.append({"index": index, "value": tokens[index]})  # index와 state 순서로 mask_lms에 넣음.
            tokens[index] = state  # 해당 토큰의 상태를 저장.
    mask_lms = sorted(mask_lms, key=lambda x: x["index"])  # 인덱스와 상태를 기존의 순서대로 넣기 위해 인덱스 순서대로 정렬.
    mask_idx = [p["index"] for p in mask_lms]  # 서브워드가 합쳐진 토큰의 인덱스가 담긴 리스트.
    mask_value = [p["value"] for p in mask_lms]  # 서브워드가 합쳐진 토큰이 담긴 리스트.

    return tokens, mask_idx, mask_value


# token 길이를 잘라내는 함수. token의 길이를 잘라내는 이유는 BERT의 계산복잡도때문이다. O(N^2)의 꼴이기 때문에 N을 줄이는 노력이 필요.
def cut_length(token1, token2, max_len):
    while 1:
        token1_len = len(token1)
        token2_len = len(token2)
        current = token1_len+token2_len
        if current < max_len:  # 목표를 이뤘으면 종료
            break
        if token1_len > token2_len:
            token1.pop(0)  # 1번째 토큰의 길이가 더 기므로 1번째 토큰의 1번째 원소를 없앰.
        else:
            token2.pop()  # 2번째 토큰의 길이가 더 기므로 2번째 토큰의 마지막 원소를 없앰.


# 단락에 있는 문장들을 pretrain data로 만드는 함수. NSP하는데 쓰인다.
def create_sentence(docs, doc_idx, doc_value, maxi, mask_prob, vocab):

    '''
    docs는 문장들이 존재. doc_idx는 docs의 인덱스들이 들어있음. doc_value는 docs의 문장이 들어있음
    maxi는 토큰의 최대 길이, mask_prob은 mask할 확률, vocab는 단어들이 들어있음.
    '''

    maxi -= 3  # CLS, SEP, SEP를 제외해야하므로 3을 빼야함.

    instances = []  # 완성된 토큰을 넣을 리스트
    cur_chunk = []  # 현재 문장이 들어갈 곳
    cur_length = 0  # 현재 문장의 길이
    last = len(doc_value)-1  # 문장의 수
    for i in range(last+1):
        cur_chunk.append(doc_value[i])  # 현재 문장을 넣음.
        cur_length += len(doc_value[i])  # 문장의 길이를 더함.
        if i == last or maxi <= cur_length:  # 마지막 문장이거나 단어가 최대 길이를 넘었으면 데이터를 생성
            if len(cur_chunk) > 0:
                a_end = 1  # 현재 문장의 길이가 1인 경우에는 고정.
                if 1 < len(cur_chunk):
                    a_end = randrange(1, len(cur_chunk))  # 문장의 길이가 1보다 큰 경우에는 랜덤하게 길이를 잡는다.
                tokens_a = []  # 1번째 문장을 저장할 곳
                for j in range(a_end):
                    tokens_a.extend(cur_chunk[j])  # a토큰에 랜덤하게 잡은 길이만큼 문장의 글자를 넣는다.

                tokens_b = []  # 2번째 문장을 저장할 곳
                if len(cur_chunk) == 1 or random() < 0.5:  # 50%의 확률로 연속되지 않은 문장을 넣는다.
                    is_next = 0
                    tokens_b_len = maxi-len(tokens_a)
                    random_doc_idx = doc_idx  # 초기값은 현재 문장의 인덱스를 준다.
                    while random_doc_idx == doc_idx :  # 현재 문장의 인덱스와 무작위 문장의 인덱스가 다르도록 처리.
                        random_doc_idx = randrange(1, len(docs))  # 무작의로 설정
                    random_doc = docs[random_doc_idx]  # 무작위 문장을 부여

                    random_fin=len(random_doc) #종료시점
                    random_start=randrange(1,random_fin) #시작 시점을 무작위로 부여

                    for j in range(random_start,random_fin):
                        tokens_b.extend(random_doc[i]) #b토큰에 랜덤하게 잡은 길이만큼 문장의 글자를 넣는다.

                else:
                    is_next=1 #문장이 연속되는 경우
                    for j in range(a_end,len(cur_chunk)):
                        tokens_b.extend(cur_chunk[j]) #토큰 a가 끝난 시점부터 문장의 끝까지 토큰b에 넣음.

                cut_length(tokens_a,tokens_b,maxi)#최대 길이를 초과한다면 길이를 잘라준다.
                assert 0<len(tokens_a)#오류처리
                assert 0<len(tokens_b)

                tokens=["[CLS]"]+tokens_a+["[SEP]"]+tokens_b+["[CLS]"] #토큰 임베딩
                segments=[0]*(len(tokens_a)+2)+[1]*(len(tokens_b)+1) #세그먼트 임베딩
                position = list(range(len(tokens)))  # 위치 임베딩

                tokens, mask_idx, mask_label=create_mask(tokens,int(len(tokens)-3)*mask_prob,vocab)  # 완성된 토큰에 mask를 입힘.

                instance={
                    "tokens":tokens,
                    "segments":segments,
                    "position":position,
                    "is_next":is_next,
                    "mask_idx":mask_idx,
                    "mask_label":mask_label
                } #현재 토큰의 특성을 나타냄.

                instances.append(instance)
            cur_chunk=[]
            cur_length=0

    return instances
