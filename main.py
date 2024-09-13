from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

sent_sim = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = sent_sim

model2_nlp = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')
# message = 'Show me a picture of Halle Berry.'
# entities = model2_nlp(message, aggregation_strategy="simple")

from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model3 = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

model3_nlp = pipeline("ner", model=model3, tokenizer=tokenizer)

qs_origin = []
Q_ents = []

qs_origin.extend([
    "Who is the director of Good Will Hunting?",
    "Who directed The Bridge on the River Kwai?",
    "Who is the director of Star Wars: Episode VI - Return of the Jedi?",
])

Q_ents.extend([
    "Good Will Hunting",
    "The Bridge on the River Kwai",
    "Star Wars: Episode VI - Return of the Jedi",
]) # should be extract by model

qs_origin.extend([
    "Who is the screenwriter of The Masked Gang: Cyprus?",
    "What is the MPAA film rating of Weathering with You?",
    "What is the genre of Good Neighbors?",
])

Q_ents.extend([
    "The Masked Gang: Cyprus",
    "Weathering with You",
    "Good Neighbors",
])

qs_origin.extend([
    "What is the box office of The Princess and the Frog?",
    "Can you tell me the publication date of Tom Meets Zizou?",
    "Who is the executive producer of X-Men: First Class?",
])

Q_ents.extend([
    "The Princess and the Frog",
    "Tom Meets Zizou",
    "X-Men: First Class",
])

qs_origin.extend([
    "Show me a picture of Halle Berry.",
    "What does Julia Roberts look like?",
    "Let me know what Sandra Bullock looks like.",
])

Q_ents.extend([
    "Halle Berry",
    "Julia Roberts",
    "Sandra Bullock",
])# TODO if many ents a question

import pandas as pd

Name_Qwiki_Qid = pd.read_csv('name_Qwiki_Qid.tsv', sep='\t')
Name_Qwiki_Qid[-5:]

names = Name_Qwiki_Qid['Str']
import numpy as np
name_embeddings = np.load("name_embeddings.npy")

import torch




def ner_test(X, y, names=names,   name_embeddings=name_embeddings, model_ner=model2_nlp, model_sent_sim=sent_sim, strategy='simple'):

    all_ents = []
    for i, q in enumerate(X):
        preds = model_ner(q, aggregation_strategy=strategy)
        if len(preds) == 0:
            print("---Try model 3---")
            preds = model3_nlp(q, aggregation_strategy=strategy)
            if len(preds) == 0:
                print(q)
                print('No name entity was found.')
                all_ents.append(q)
            else:
                all_ent = q[preds[0]['start']: preds[-1]['end']]
                all_ents.append(all_ent)
        else:
            all_ent = q[preds[0]['start']: preds[-1]['end']]
            # remove substring 'xxx rating' done
            all_ents.append(all_ent)

    all_ents_embeddings = model_sent_sim.encode(all_ents)
    sims = util.pytorch_cos_sim(torch.tensor(all_ents_embeddings), torch.tensor(name_embeddings))
    idx = torch.max(sims, 1).indices
    
    ner_pred = all_ents
    y_pred = np.array([names[int(id)] for id in idx])
    print(y_pred)
    y = np.array(y)
    for i in range(len(y)):
        if y_pred[i]!=y[i]:
            # print("-----------------------------")
            print("Question: {}".format(X[i]))
            print("NER prediction: {}".format(ner_pred[i]))
            print("NER matching: {}".format(y_pred[i]))
            print("Ground truth: {}".format(y[i]))
            print("-----------------------------")
            # print(X[i], ner_pred[i], y_pred[i], y[i], )
    accuracy = np.mean(y==y_pred)
    return accuracy

# small_data = pd.read_csv('Datasets/Relations_small_X_y.tsv', sep='\t')
# i, step = 0, 30
# acc = ner_test(list(small_data['X'][i:i+step]), list(small_data['y'][i:i+step]))
# acc = ner_test(["What is the MPAA film rating of Weathering with You?",], ["Weathering with You"])
# print(acc)

def ner_app(q, names=names,   name_embeddings=name_embeddings, model_ner=model2_nlp, model_sent_sim=sent_sim, strategy='simple'):

    pred_ent = ''
    
    preds = model_ner(q, aggregation_strategy=strategy)
    if len(preds) == 0:
        print("---Try model 3---")
        preds = model3_nlp(q, aggregation_strategy=strategy)
        if len(preds) == 0:
            print(q)
            print('No name entity was found.')
            pred_ent = q
        else:
            pred_ent = q[preds[0]['start']: preds[-1]['end']]
    else:
        pred_ent = q[preds[0]['start']: preds[-1]['end']]

    all_ents_embeddings = model_sent_sim.encode(pred_ent)
    sims = util.pytorch_cos_sim(torch.tensor(all_ents_embeddings), torch.tensor(name_embeddings))
    idx = torch.max(sims, 1).indices
    # print(int(idx))
    y_pred = names[int(idx[0])]

    if y_pred in q:
        q = q.replace(y_pred, '<>')
    
    return y_pred, q


# Q, mask_q = ner_app("What is the MPAA film rating of Weathering with You?")
# print(Q, mask_q)

def ner_for_recommend(q, names=names,   name_embeddings=name_embeddings, model_ner=model2_nlp, model_sent_sim=sent_sim, strategy='simple'):

    Qs = []
    preds = model_ner(q, aggregation_strategy=strategy)
    if len(preds) == 0:
        print("---Try model 3---")
        preds = model3_nlp(q, aggregation_strategy=strategy)
        if len(preds) == 0:
            print(q)
            print('No name entity was found.')
            ents = [q]
        else:
            ents = [pred['word'] for pred in preds ]
    else:
        ents = [pred['word'] for pred in preds ]
    

    ents_embeddings = model_sent_sim.encode(ents)
    sims = util.pytorch_cos_sim(torch.tensor(ents_embeddings), torch.tensor(name_embeddings))
    idx = torch.max(sims, 1).indices
    # print(idx)
    for id in np.array(idx):
        Qs.append(names[id])

    return Qs

# print(ner_for_recommend("Recommend movies similar to Hamlet and Othello."))
# print(ner_for_recommend("Recommend movies like Nightmare on Elm Street, Friday the 13th, and Halloween."))
# print(ner_for_recommend("Given that I like The Lion King, Pocahontas, and The Beauty and the Beast, can you recommend some movies?"))

import yake
kw_extractor = yake.KeywordExtractor()
# text = """spaCy is an open-source software library for advanced natural language processing, written in the programming languages Python and Cython. The library is published under the MIT license and its main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion."""
# text = """Who is the director of ?"""
# text = """Who directed ?"""
# text = """What is the MPAA film rating of?"""
# text = """Let me know what  looks like."""
# text = """Who directed ?"""
# text = """Let me know what  looks like"""
text = """Show me a picture of ."""
language = "en"
max_ngram_size = 8
deduplication_threshold = 0.9
numOfKeywords = 1
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)

import pickle
with open('P2label.pickle', 'rb') as handle:
    P2label = pickle.load(handle)
with open('label2P.pickle', 'rb') as handle:
    label2P = pickle.load(handle)
labels = list(P2label.values())

Name2Qid = dict(zip(Name_Qwiki_Qid['Str'], Name_Qwiki_Qid['Qid']))
Qid2Name = dict(zip(Name_Qwiki_Qid['Qid'], Name_Qwiki_Qid['Str']))

labels = list(P2label.values())

special_conditions_reverse = {'description': '<http://schema.org/description>',
       'label': '<http://www.w3.org/2000/01/rdf-schema#label>',
       'tag': '<http://ddis.ch/atai/tag>', 'rating': '<http://ddis.ch/atai/rating>'}
extension_common = ['description', 'label', 'tag', 'rating']
extension_recommend = ['recommend']
extension_multimedia = ['look like'] # seem; appear; look like; look; resemble; bear resemblance to; be like.
labels.extend(extension_common)
labels.extend(extension_recommend)
labels.extend(extension_multimedia)

label_embeddings = model.encode(labels)

# def label_prediction(questions, use_keywords=True, use_expanding=False):
#     keywords = questions
#     if use_keywords:
#         keywords = []
#         for q in questions:
#             keyword = custom_kw_extractor.extract_keywords(q)
#             if not keyword:
#                 print("Use whole question")
#                 keywords.append(q)
#             else:
#                 print("Keyword: ", keyword[0][0])
#                 keywords.append(keyword[0][0])

#         print("Keywords: ", keywords)
    
#     question_embeddings = model.encode(keywords)
#     # print(question_embeddings)
#     preds = []
#     for embd in question_embeddings:
#         results = [util.pytorch_cos_sim(embd, label) for label in label_embeddings]
#         pred = labels[results.index(max(results))]
#         preds.append(pred)
#     return preds

def label_prediction(question, use_keywords=True, use_expanding=False):

    if 'ecommend' in question:
        return 'recommend'
    
    keywords = [question]
    if use_keywords:
        keywords = []
        keyword = custom_kw_extractor.extract_keywords(question)
        if not keyword:
            print("Use whole question")
            keywords.append(question)
        else:
            print("Keyword: ", keyword[0][0])
            keywords.append(keyword[0][0])

        print("Keywords: ", keywords)
    
    question_embeddings = model.encode(keywords)
    # print(question_embeddings)

    sims = util.pytorch_cos_sim(torch.tensor(question_embeddings), torch.tensor(label_embeddings))
    idx = torch.max(sims, 1).indices
    y_pred = labels[int(idx[0])]

    return y_pred

from rdflib.namespace import Namespace, RDF, RDFS, XSD
from rdflib.term import URIRef, Literal
import csv
import json
import networkx as nx
import rdflib
import pywikibot

graph = rdflib.Graph()
graph.parse('Datasets/14_graph.nt', format='turtle')


def Qwiki2name_func(Qwiki: str):
    if Qwiki == None:
        return None
    if not Qwiki.startswith('http'):
        Qwiki = 'http://www.wikidata.org/entity/{}'.format(Qwiki)
    sql = \
        """
    prefix wdt: <http://www.wikidata.org/prop/direct/> 
    prefix wd: <http://www.wikidata.org/entity/> 
    SELECT ?lbl
    WHERE{
    <%s> rdfs:label ?lbl.
    FILTER(LANG(?lbl)='en')
    }

    """ % (Qwiki)

    tl = list(graph.query(sql))
    if len(tl)>0:
        for t in tl:
            for m in t:
                print(str(m))
                return str(m)
    else:
        return None

def name2Qid_func(name: str, func=1):
    if name == None:
        return None
    if func==1:
    
        sql1 = \
        """
    prefix wdt: <http://www.wikidata.org/prop/direct/> 
    prefix wd: <http://www.wikidata.org/entity/> 
    SELECT ?Qid
    WHERE{
    ?Qid rdfs:label "%s"@en .
    }

    """ % (name)

        sql2 = \
        """
    prefix wdt: <http://www.wikidata.org/prop/direct/> 
    prefix wd: <http://www.wikidata.org/entity/> 
    SELECT ?Qid
    WHERE{
    ?Qid rdfs:label "%s"@en.
    ?Qid wdt:P31/wdt:P279* wd:Q11424
    
    }

    """ % (name)
    
    

        Qids = []
        tl = list(graph.query(sql1))
        for t in tl:
            for m in t:
                Qids.append(str(m).removeprefix("http://www.wikidata.org/entity/"))
        # print(len(Qids))
        if len(Qids) == 0:
            print('Name2Qid failed, name: {}'.format(name))
            return None
        elif len(Qids) == 1:
            return Qids[0]
        
        elif len(Qids) > 1:
            Qids2 = []
            tl2 = list(graph.query(sql2))
            # print("TL:", tl2)
            for t in tl2:
                for m in t:
                    Qids2.append(str(m).removeprefix("http://www.wikidata.org/entity/"))
            if len(Qids2) >= 1:
                return np.random.choice(Qids2)
            else: 
                # print(Qids)
                return np.random.choice(Qids)
        
    else:
        if name in Name2Qid.keys():
            # "Star Wars: Episode VI - Return of the Jedi"
            return Name2Qid[name]
        else:
            print('Name2Qid failed, name: {}'.format(name))
            return None

def label2Pid_func(label):
    if label in extension_multimedia:
        return label2P['image']

    if label in label2P.keys():
        return label2P[label]
    else:
        print('Label2Pid failed.')
        return None


recommend_labels = ['recommend']
multimedia_labels = ['image', 'look like']
special_conditions_reverse = {'description': '<http://schema.org/description>',
       'label': '<http://www.w3.org/2000/01/rdf-schema#label>',
       'tag': '<http://ddis.ch/atai/tag>', 'rating': '<http://ddis.ch/atai/rating>'}

def distinguish_question_type(pred):
    question_type = 'common'
    if pred in special_conditions_reverse.keys():
        question_type = 'special'
        
    elif pred in recommend_labels:
        question_type = 'recommend'

    elif pred in multimedia_labels:
        question_type = 'multimedia'
    return question_type

def answer_special_questions(Q, P):
    Qid = name2Qid_func(Q)
    Puri = special_conditions_reverse[P]
    common_sql = \
"""
prefix wdt: <http://www.wikidata.org/prop/direct/> 
prefix wd: <http://www.wikidata.org/entity/> 
SELECT ?item
WHERE{
  wd:%s %s ?item .
}
""" % (Qid, Puri)
    results = [str(s) for s, in graph.query(common_sql)]
    if len(results):
      # print(common_sql)
      return results[0]
    else:
      print('Answer special questions failed, sql: {}'.format(common_sql))
      return None

def answer_factual_questions(Q, P, func='fac'):

    Qid = name2Qid_func(Q)
    Pid = label2Pid_func(P)
    common_sql = \
"""
prefix wdt: <http://www.wikidata.org/prop/direct/> 
prefix wd: <http://www.wikidata.org/entity/> 
SELECT ?label
WHERE{
  wd:%s wdt:%s ?item .
  ?item rdfs:label ?label .
  FILTER(LANG(?label) = "en").
}
""" % (Qid, Pid)
    results = [str(s) for s, in graph.query(common_sql)]
    if len(results):
      # print(common_sql)
      return results[:3]
    else:
      print('Answer factual questions failed.'.format(common_sql))
      return None

from sklearn.metrics import pairwise_distances
WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
DDIS = rdflib.Namespace('http://ddis.ch/atai/')
RDFS = rdflib.namespace.RDFS
SCHEMA = rdflib.Namespace('http://schema.org/')

entity_emb = np.load('Datasets/EmbeddingData/entity_embeds.npy')
relation_emb = np.load('Datasets/EmbeddingData/relation_embeds.npy')

with open('Datasets/EmbeddingData/entity_ids.del', 'r') as ifile:
    ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
    id2ent = {v: k for k, v in ent2id.items()}
with open('Datasets/EmbeddingData/relation_ids.del', 'r') as ifile:
    rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
    id2rel = {v: k for k, v in rel2id.items()}

ent2lbl = {ent: str(lbl) for ent, lbl in graph.subject_objects(RDFS.label)}
lbl2ent = {lbl: ent for ent, lbl in ent2lbl.items()}

# def answer_embedding_questions(Q, P, func='emb'):

#     Qid = name2Qid_func(Q)
#     Pid = label2Pid_func(P)
#     head = entity_emb[ent2id[WD[Qid]]]
#     pred = relation_emb[rel2id[WDT[Pid]]]
#     # add vectors according to TransE scoring function.
#     lhs = head + pred
#     # compute distance to *any* entity
#     dist = pairwise_distances(lhs.reshape(1, -1), entity_emb).reshape(-1)
#     # find most plausible entities
#     most_likely = dist.argsort()

#     print([(id2ent[idx][len(WD):], ent2lbl[id2ent[idx]], dist[idx], i) for i, idx in enumerate(most_likely[:3])])
#     emb_result = ent2lbl[id2ent[most_likely[0]]]

#     return emb_result

def answer_embedding_questions(Q, P, func='emb'):
    
    Qid = name2Qid_func(Q)
    Pid = label2Pid_func(P)

    if (WD[Qid] not in ent2id.keys()) or (WDT[Pid] not in rel2id.keys()):
        return None
    head = entity_emb[ent2id[WD[Qid]]]
    pred = relation_emb[rel2id[WDT[Pid]]]
    # add vectors according to TransE scoring function.
    lhs = head + pred
    # compute distance to *any* entity
    dist = pairwise_distances(lhs.reshape(1, -1), entity_emb).reshape(-1)
    # find most plausible entities
    most_likely = dist.argsort()

    print([(id2ent[idx][len(WD):], ent2lbl[id2ent[idx]], dist[idx], i) for i, idx in enumerate(most_likely[:3])])
    emb_result = ent2lbl[id2ent[most_likely[0]]]

    return emb_result

cro_df = pd.read_csv('Datasets/CrowdData/Clean_Cro.tsv', sep='\t')

def answer_crowdsourcing_questions(Q, P, func='cro', use_id=False):

    
    if use_id:
        Qid = Q
        Pid = P
    else:
        Qid = name2Qid_func(Q)
        Pid = label2Pid_func(P)

    # print(Qid, Pid)
    
    micro_task = cro_df.loc[(cro_df['Input1ID']=='wd:{}'.format(Qid)) & (cro_df['Input2ID']=='wdt:{}'.format(Pid))]
    if len(micro_task) > 0:
        #   GroupAnswers | TrueCount | FalseCount | Agreement
        #   FixPosition | FixValue 
        #   Input3ID
        GroupAnswers, TrueCount, FalseCount, Agreement, FixPositions, FixValues, Input3ID = (
        list(micro_task['GroupAnswers'])[0], 
        list(micro_task['TrueCount'])[0],
        list(micro_task['FalseCount'])[0],
        list(micro_task['Agreement'])[0],
        list(micro_task['FixPosition']),
        list(micro_task['FixValue']),
        str(list(micro_task['Input3ID'])[0]),
        )
        print(Input3ID)
        response_type = 'correct'
        if GroupAnswers == True:
            # use the provided answer

            if not Input3ID.startswith('wd:'):
                # return string
                cro_result = Input3ID
            else:
                # transfer id to label
                Qid = Input3ID[3:]
                cro_result = Qwiki2name_func(Qid)
        else:
           

            # fix or mention
            fixable = micro_task[micro_task['FixPosition'].str.contains('Object') & (~micro_task['FixValue'].str.contains('Empty'))]
            # if FixPositions != 'Empty' and FixValues != 'Empty':
            # both_not_empty = micro_task[(~micro_task['FixPosition'].str.contains('Empty')) & (~micro_task['FixValue'].str.contains('Empty'))]

            if len(fixable) > 0:
                fix_values = fixable['FixValue'].unique()
                value = fix_values[0]
                # fix
                response_type = 'fix'
                if value.startswith('wd:'):
                    value = value[3:]
                if value.startswith('Q'):
                    cro_result = Qwiki2name_func(value)
                else:
                    cro_result = value
                
                
            else:
                if Input3ID[:3] != 'wd:':
                    # return string
                    cro_result = Input3ID
                else:
                    # transfer id to label
                    Qid = Input3ID[3:]
                    cro_result = Qwiki2name_func(Qid)
                # just mention
                # The answer in knowledge graph is wrong -- according to the crowd, ~~~
                response_type = 'mention'

                

        return {
        'response_type': response_type,
        'result': cro_result,
        'TrueCount': TrueCount, 
        'FalseCount': FalseCount,
        'Agreement': Agreement,
        }

    else:
        print('what')
        return None



def answer_common_questions(Q, P, func='cro'):
    # fuc: fac, emb, cro
    answer = answer_crowdsourcing_questions(Q, P)
    if answer:
      return {'type': 'cro', 'answer': answer}
    else:
      answer_fac = answer_factual_questions(Q, P)
      answer_emb = answer_embedding_questions(Q, P)
      if answer_fac == None and answer_emb == None:
        return None
      elif answer_fac != None and answer_emb == None:
        return {'type': 'fac', 'answer': answer_fac}
      elif answer_fac == None and answer_emb != None:
        return {'type': 'emb', 'answer': answer_emb}
      else:
        return {'type': 'fac and emb', 'answer': (answer_emb, answer_fac)}
      
with open('Datasets/ImageData/id2Code.pickle', 'rb') as handle:
    id2Code = pickle.load(handle)

def answer_multimedia_questions(Q, P):
    Qid = name2Qid_func(Q)
    print(Qid)

    sql = \
        """
    prefix wdt: <http://www.wikidata.org/prop/direct/> 
    prefix wd: <http://www.wikidata.org/entity/> 
    SELECT ?id
    WHERE{
    wd:%s wdt:P345 ?id .
    }

    """ % (Qid)

    id = ''
    tl = list(graph.query(sql))
    for t in tl:
        for m in t:
            id = str(m)
            break
    print(id)
    if id in id2Code:
        return "image:{}".format(id2Code[id])
    else:
        print("Id is not in id to code dictionary .")
        return None

def search_recommendations(Qs, map, Pids_withpre):
    print(map)
    # genres, public_dates, directors, set_in_period, tags
    
    # lines_genre = "\n".join(["?movie %s wd:%s ." % (Pids_withpre[0], name2Qid_func(list(map['genre'])[i])) for i in range(len(list(map['genre'])))]) if map['genre'] else ''
    lines_genre = "?movie %s <%s> ." % (Pids_withpre[0], map['genre']) if map['genre'] else ''


    lines_publication_date = """
    ?movie %s ?pubdate .
    FILTER (?pubdate >= "%s-01-01"^^xsd:date)
    FILTER (?pubdate < "%s-12-31"^^xsd:date)
    """ % (Pids_withpre[1], list(map['publication_date'])[0], list(map['publication_date'])[1]) if map['publication_date'] else ''
    
    # lines_director = "\n".join(["?movie %s wd:%s ." % (Pids_withpre[2], name2Qid_func(list(map['director'])[i])) for i in range(len(list(map['director'])))]) if map['director'] else ''
    lines_director = "?movie %s <%s> ." % (Pids_withpre[2], map['director']) if map['director'] else ''


    # lines_period = """
    # ?movie wdt:P2408 ?period .
    # ?period rdfs:label "%s"@en.

    # """ % (list(map['set_in_period'])[0]) if map['set_in_period'] else ''

    lines_period = """
    ?movie wdt:P2408 <%s> .

    """ % (map['set_in_period']) if map['set_in_period'] else ''
    
    # lines_tag = "\n".join(["?movie %s wd:%s ." % (Pids_withpre[4], name2Qid_func(list(map['tag'])[i])) for i in range(len(list(map['tag'])))]) if map['tag'] else ''

    def genreate_sql(lines_genre, lines_publication_date, lines_director, lines_period):
        sql = \
            """
        prefix wdt: <http://www.wikidata.org/prop/direct/> 
        prefix wd: <http://www.wikidata.org/entity/> 
        prefix ps: <http://www.wikidata.org/prop/statement/>
        prefix p: <http://www.wikidata.org/prop/>
        SELECT ?lb1
        WHERE{
        
        %s\n%s\n%s\n%s

        ?movie wdt:P31/wdt:P279* wd:Q11424 .
        ?movie rdfs:label ?lb1 .
        FILTER(LANG(?lb1) = "en").
        }
        LIMIT 6
        """ % (lines_genre, lines_publication_date, lines_director, lines_period)
        # print(sql)
        return(sql)
    sql = genreate_sql(lines_genre, lines_publication_date, lines_director, lines_period)
    # print(sql)
    movies = set([str(s) for s, in graph.query(sql)])
    movies = movies.difference(set(Qs))

    if len(movies) == 0:
        sql = genreate_sql(lines_genre, lines_publication_date, lines_director, '')
        movies = set([str(s) for s, in graph.query(sql)])
        movies = movies.difference(set(Qs))
        if len(movies) == 0:
            sql = genreate_sql(lines_genre, lines_publication_date, '', '')
            movies = set([str(s) for s, in graph.query(sql)])
            movies = movies.difference(set(Qs))
            if len(movies) == 0:
                sql = genreate_sql(lines_genre, '', '', '')
                movies = set([str(s) for s, in graph.query(sql)])
                movies = movies.difference(set(Qs))

    
    # print(movies)
    if len(movies) > 0:
        if len(movies) > 3:
            return list(movies)[:3]
        else:
            return list(movies)
    else:
        return None
    # return list(movies)[:3] if len(movies) > 0 else None

def fetch_features_from_sql(Qid, Pids_withpre):
    sqls = []
    for i, Pid in enumerate(Pids_withpre):
      sql = \
          """
    prefix wdt: <http://www.wikidata.org/prop/direct/> 
    prefix wd: <http://www.wikidata.org/entity/> 
    SELECT ?item1
    WHERE{
    wd:%s %s ?item1 .
    ?item1 rdfs:label ?lb1 .
    FILTER(LANG(?lb1) = "en").
    }
    """ % (Qid, Pid)
    # lb1
      sqls.append(sql)

    sqls[1] = \
          """
    prefix wdt: <http://www.wikidata.org/prop/direct/> 
    prefix wd: <http://www.wikidata.org/entity/> 
    SELECT ?lb1
    WHERE{
    wd:%s %s ?item1 .
    ?item1 rdfs:label ?lb1 .
    FILTER(LANG(?lb1) = "en").
    }
    """ % (Qid, Pids_withpre[1])
    genres = [str(s) for s, in graph.query(sqls[0])]
    public_dates = [str(s) for s, in graph.query(sqls[1])]
    directors = [str(s) for s, in graph.query(sqls[2])]
    set_in_period = [str(s) for s, in graph.query(sqls[3])]
    tags = [str(s) for s, in graph.query(sqls[4])]


    # print("{}\n{}\n{}\n{}\n{}\n".format(genres, public_dates, directors, set_in_period, tags))
    return genres, public_dates, directors, set_in_period, tags

from collections import Counter

def analysis_recommendation(features_all):
    # Ps = ['genre', 'publication date', 'director', 'set in period'] + tag

    genres, public_dates, directors, set_in_period, tags = features_all
    def analysis_intersection(features):
        if len(features) == 0:
            return None
        intersection = set(features[0])
        for l in features[1:]:
            intersection = intersection & set(l)
        return list(intersection)[0] if len(intersection) > 0 else None

    def analysis_min_max(features):
        if len(features) == 0:
            return None
        union = set(features[0])
        for l in features[1:]:
            union = union | set(l)
        if len(union) == 0:
            return None
        dates = [int(date[:4]) for date in union]
        return min(dates), max(dates)

    def analysis_most_common(features):
        if len(features) == 0:
            return None
        li = features[0]
        for l in features[1:]:
            li.extend(l)
        if len(li) == 0:
            return None
        
        # for i, period in enumerate(li):
        #     if period[-1:] != 's':
        #         li[i] = str((int(period) // 10) * 10)+'s'
        # print(li)
        counter = Counter(li)
        # print(counter.most_common(1)[0][0])
        return counter.most_common(1)[0][0]

    analysis = analysis_most_common(genres), \
            analysis_min_max(public_dates), \
                analysis_intersection(directors), \
                    analysis_most_common(set_in_period), \
                        analysis_intersection(tags)

    
    return analysis


    # print(analysis_public_dates(['2018-08','2020-02', '2010-02', '2019']))
    
    return None

def answer_recommend_questions(Qs: list):
  
    Qids = [name2Qid_func(Q) for Q in Qs]
    print("Qids:", Qids)
    Ps = ['genre', 'publication date', 'director', 'set in period']
    Pids_withpre = ['wdt:{}'.format(label2Pid_func(P)) for P in Ps]
    tag_rdf = special_conditions_reverse['tag']
    Pids_withpre.append(tag_rdf)


    features_all = [[] for i in range(len(Pids_withpre))]
    for i, Qid in enumerate(Qids):
      features = fetch_features_from_sql(Qid, Pids_withpre)
      for j, feature in enumerate(features):
        features_all[j].append(feature)

    analysis = analysis_recommendation(features_all)

    recommend_map = {}
    recommend_map.update({"director": analysis[2]})
    recommend_map.update({"set_in_period": analysis[3]})
    recommend_map.update({"genre": analysis[0]})
    recommend_map.update({"publication_date": analysis[1]})
    
    # print(analysis_results)
    recommendations = search_recommendations(Qs, recommend_map, Pids_withpre)

    map = recommend_map
    map['director'] = Qwiki2name_func(map['director'])
    map['set_in_period'] = Qwiki2name_func(map['set_in_period'])
    map['genre'] = Qwiki2name_func(map['genre'])

      
    return recommend_map, recommendations

def answer_question(question_type, **kwargs):
    if question_type == 'common':
        return answer_common_questions(kwargs['Q'], kwargs['P'])
    elif question_type == 'recommend':
        return answer_recommend_questions(kwargs['Qs']) #list
    elif question_type == 'multimedia':
        return answer_multimedia_questions(kwargs['Q'], kwargs['P'])
    elif question_type == 'special':
        return answer_special_questions(kwargs['Q'], kwargs['P'])

def full_answer_generation_fai(answer):
    # Failed to answer question
    choices = \
    [
        "I'm sorry that I can't answer this question to you now, can you change another one?",
        "This question is a bit of hard for me, ask me another one please!",
        "Oops! It's difficult for me to show you answer to this question, ask me a new one!",
    ]

    return np.random.choice(choices)

def full_answer_generation_mul(answer):
    # Image code
    return answer

def full_answer_generation_rec(answer):
    # Failed to answer question
    map, rec = answer
    choices_1 = \
    [
        "I would recommend you these movies: {}".format(', '.join(rec)),
        "I would like to recommend these movies to you: {}".format(', '.join(rec)),
        "I suppose you will also like: {}".format(', '.join(rec)),
    ]

    
    
    choice_genre = "in the {} genre".format(map['genre']) if map['genre'] else None
    choice_director = "directed by {}".format(map['director']) if map['director'] else None
    if map["publication_date"]:
        if map["publication_date"][0] == map["publication_date"][1]:
            choice_publicated = "pulicated around {}".format(map["publication_date"][0])
        else:
            choice_publicated = "pulicated from {} to {}".format(map["publication_date"][0], map["publication_date"][1])
    else:
        choice_publicated = None
    
    choice_period = "with a background period around {}".format(map["set_in_period"]) if map["set_in_period"] else None

    merge = []
    if choice_genre:
        merge.append(choice_genre)
    if choice_director:
        merge.append(choice_director)
    if choice_publicated:
        merge.append(choice_publicated)
    if choice_period:
        merge.append(choice_period)
    other = ', '.join(merge)

    choices_2 = \
        [
            ". By aspects, movies {} may speak to you.".format(other),
            ". Besides, movies {} may to your appetite.".format(other),
            ", and you may have interest in movies {}.".format(other),
        ]

    c1 = np.random.choice(choices_1)
    c2 = np.random.choice(choices_2)

    return "{}{}".format(c1, c2)

def full_answer_generation_spe(answer):
    # special

    # choices = \
    #     [
    #         "The answer to this question is {}.".format(answer),
    #         "I know this, {}.".format(answer),
    #         "Good question, I suppose the answer is {}.".format(answer),
    #         "Hmm, a hard question, but I know the answer is {}".format(answer),
    #         "I support {} is the answer you want.".format(answer),
    #     ]

    choices = \
        [
            "The answer to this question is: {}.".format(answer),
            "I know this, {}.".format(answer),
            "Good question, I suppose the answer is: {}.".format(answer),
            "Hmm, a hard question, but I know the answer: {}".format(answer),
            "I support {} is the answer you want.".format(answer),
        ]

    return np.random.choice(choices)

def full_answer_generation_fac(answer):
    # factual

    if len(answer) == 1:
        answer = answer[0]


        choices = \
            [
                "The answer to this question is {}.".format(answer),
                "I know this, {}.".format(answer),
                "Good question, I suppose the answer is {}.".format(answer),
                "Hmm, a hard question, but I know the answer is {}".format(answer),
                "I support {} is the answer you want.".format(answer),
            ]
    elif len(answer) == 2:

        choices = \
            [
                "The answers to this question are: {}.".format(' and '.join(answer)),
                "I know this, {}.".format(' and '.join(answer)),
                "Good question, I suppose the answers will be: {}.".format(' and '.join(answer)),
                "Hmm, a hard question, but I know the answers: {}".format(' and '.join(answer)),
                "I support {} are the answers you want.".format(' and '.join(answer)),
            ]
    
    elif len(answer) > 2:

        choices = \
            [
                "The answers to this question are: {}.".format(', '.join(answer)),
                "I know this, {}.".format(', '.join(answer)),
                "Good question, I suppose the answers will be: {}.".format(', '.join(answer)),
                "Hmm, a hard question, but I know the answers: {}".format(', '.join(answer)),
                "I support {} are the answers you want.".format(', '.join(answer)),
            ]

    return np.random.choice(choices)

def full_answer_generation_emb(answer):
    # embedding

    choices = \
        [
            "According to embeddings, the answer to this question is {}.".format(answer),
            "I find the answer to this question from embeddings: {}.".format(answer),
            "Good question, I suppose the answer is {}, suggested by embeddings.".format(answer),
            "Hmm, a hard question! But I know the answer is likely to be {}, provided by embeddings.".format(answer),
            "Take the embeddings as reference, I support {} is the answer you want.".format(answer),
        ]
    

    return np.random.choice(choices)

def full_answer_generation_bot(answer):
    # factual and embedding
    # {'type': 'fac and emb', 'answer': [answer_emb, answer_fac]}

    answer_emb, answer_fac = answer
    full_emb = full_answer_generation_emb(answer_emb)

    if len(answer_fac) == 1:
        answer_fac = answer_fac[0]
        
        if answer_emb == answer_fac:
            full_fac = np.random.choice([
                "The answer is the same if I find the answer in knowledge graph.",
                "I also get the same answer from knowledge graph, what a coincidence!",
            ])
            

        else:
            full_fac = np.random.choice([
            "However, I find another possible answer in knowledge graph: {}".format(answer_fac),
            "However, the answer is different if I search it in knowledge graph: {}.".format(answer_fac),
            "Different answer would be fetched from knowledge graph: {}.".format(answer_fac)
        ])

        
    else:
        full_fac = np.random.choice([
                "However, there would be multiple answers in knowledge graph: {}.".format(', '.join(answer_fac)),
                "While multiple answers would be fetched from knowledge graph: {}.".format(', '.join(answer_fac)),
            ])


       

    return "{} {}".format(full_emb, full_fac)

def full_answer_generation_cro(answer: dict):
    # crowdsourcing
    # {
        # 'response_type': response_type, # correct fix mention
        # 'result': cro_result, # label
        # 'TrueCount': TrueCount, # int
        # 'FalseCount': FalseCount, # int
        # 'Agreement': Agreement, # float
        # }

    # London - according to the crowd, who had an inter-rater agreement of 0.72 in this batch. 
    # The answer distribution for this specific task was 2 support votes and 1 reject vote. 

    for key in answer.keys():
        if answer[key] == None:
            return full_answer_generation_fai(None)

    choices_2 = [
            "The crowd had an inter-rater agreement of {}, in the related batch. ".format(answer['Agreement']) + 
            "The answer distribution of this micro task was {} support and {} reject.".format(answer['TrueCount'], answer['FalseCount']),
            "The inter-rater agreement of relevant batch is {}. ".format(answer['Agreement']) + 
            "The concrete distribution of relevant micro task was {} support and {} reject.".format(answer['TrueCount'], answer['FalseCount']),
        ]
    info_answer = np.random.choice(choices_2)

    if answer['response_type'] == 'correct':
        part_answer = np.random.choice([
            "Hmm, a hard question! But I know the answer is probably to be {}, provided by crowd.".format(answer['result']),
            "Good question. I suppose the answer is {}, suggested by crowd.".format(answer['result']),
            "According to crowd, the answer to this question is {}.".format(answer['result']),
            "I find the answer to this question from crowd: {}.".format(answer['result']),
            "The crowd suggested that the answer is: {}".format(answer['result'])
        ])

        
    elif answer['response_type'] == 'fix':
        part_answer = np.random.choice([
            "Hmm, a hard question! But I know the answer is probably to be {}, which is fixed by crowd.".format(answer['result']),
            "Good question. I suppose the answer is {}, fixed by crowd.".format(answer['result']),
            "According to crowd, the fixed answer to this question is {}.".format(answer['result']),
            "I find a fixed answer to this question from crowd: {}.".format(answer['result']),
            "The crowd suggested that the fixed answer is: {}".format(answer['result'])
        ])

    else:
        # mention
        part_answer = np.random.choice([
            "Hmm, a hard question! The answer is probably to be {} from knowledge graph. However, the crowd suggested that this relationship is wrong. Sorry that I don't have complete information. But I can show you some other information.".format(answer['result']),
            "Good question. The answer in knowledge graph is {}, but the crowd think it's wrong. I don't have enough information to fix it. Here is some other analysis:".format(answer['result']),
            "According to crowd, the answer to this question is wrong, although we can get it from knowledge graph: {}. Sorry that I don't have complete information. Here is some other information I want to show you.".format(answer['result']),
        ])
    
    

    return "{} {}".format(part_answer, info_answer)

def full_answer_generation(question_type, answer):
    if not answer:
        return full_answer_generation_fai(answer)
    if question_type == 'multimedia':
        return full_answer_generation_mul(answer)
    elif question_type == 'recommend':
        return full_answer_generation_rec(answer)
    elif question_type == 'special':
        return full_answer_generation_spe(answer)
    elif question_type == 'common':
        if answer['type'] == 'cro':
            return full_answer_generation_cro(answer['answer'])
        elif answer['type'] == 'emb':
            return full_answer_generation_emb(answer['answer'])
        elif answer['type'] == 'fac':
            return full_answer_generation_fac(answer['answer'])
        elif answer['type'] == 'fac and emb':
            return full_answer_generation_bot(answer['answer'])
    return full_answer_generation_fai(answer)






