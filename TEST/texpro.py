
import pandas as pd
from pandas import ExcelWriter
import nltk
import re
import os
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize,word_tokenize
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
import spacy
nlp = spacy.load('en_core_web_sm')
NUM_CLUSTERS = 5
simidoc = []
simdoc = []
fsent = []
sents = []
sendf = pd.DataFrame(columns = ["Recommendations"])
checklists = []
finsent = []
upsent =[]
ctb = []
constraints = ["must","should","shouldn't","important","Remember","noted","cautious","make sure",'need','recommended']
constraints=[x.lower() for x in constraints]
befwords = ['This','That','These','Then','However','Therefore','Then','Instead','them','After','Next','Hence','Otherwise','Additionally',r'\bFor example\b',r'\bFor instance\b']
nextcons = [r'\bFor example\b',r'\bFor instance\b','they','This']
aftwords = ['following','?','Next','After','This']
bcorefs=[x.lower() for x in befwords]
acorefs=[x.lower() for x in aftwords]
skip = [r'\bfollowing diagrams\b',r'\bfollowing snippet\b',r'\bfollowing code\b',r'\bthis lesson\b',r'\bFigure\b',r'\bThis page\b',r'\bThis lesson\b',r'\bThis guide\b']
strong_words = ['^Note:.*', '^Caution:.*', '^Tip:.*', '^Warning.*']
strgcorefs = ['This','However','above','Hence','this','Therefore']
desc = []
strongtexts = []
df1 = pd.DataFrame()


def kmeans(df1):
    des = df1.values
    for sen in des:
        for s in sen:
            desc.append(s)
    tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    text = tfidf.fit_transform(desc)
    words = tfidf.get_feature_names_out()

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = 8, n_init = 1, tol = 0.01, max_iter = 200)
    #fit the data
    labels = kmeans.fit_predict(text)
    for index,sent in enumerate(desc):
        print(str(labels[index]) + ":" + str(sent))

def checkpunct(doc):
  for tok in doc:
    if tok.text == ',' and 'punct' in tok.dep_:
      return 0

def checkprevcond(prevs,doc,ind,df):
    bc = [b for b in bcorefs if b in prevs.lower()]
    skp = [f for f in skip if re.search(f, prevs)]
    if bc !=[] and skp == []:
        if ind != 0:
            newprev, ind = checkprev(df, ind)
            newsent = newprev + prevs + doc.text
        else:
            newsent = prevs + doc.text
        return newsent

    else:
        newsent = prevs + doc.text
        return newsent

def checkbfconst(df,doc,index):
    for tok in doc:
        if tok.text == 'this' and (tok.dep_ == 'det' or tok.dep_ == 'pobj' or tok.dep_ == 'dobj'):
            prev,ind = checkprev(df, index)
            if prev not in finsent:
                fullsent = checkprevcond(prev,doc,ind,df)
                return fullsent
        elif tok.text == 'This':
            if doc[0].text == tok.text:
                prev, ind = checkprev(df, index)
                fullsent = checkprevcond(prev, doc, ind, df)
                return fullsent

        elif tok.text == 'this' and (tok.dep_ == 'nsubj'):
            punct = checkpunct(doc)
            if punct == 0:
                return doc.text
            else:
                prev,ind = checkprev(df, index)
                fullsent = prev + doc.text
                return fullsent
        elif tok.text.lower() == 'that' and (tok.dep_ == 'det'):
            prev, ind = checkprev(df, index)
            fullsent = prev + doc.text
            return fullsent
        elif tok.text.lower() == 'that' and (tok.dep_ == 'mark' or tok.dep_ == 'nsubj' or tok.dep_ == 'nsubjpass'):
            return doc.text
        elif tok.text.lower() == 'instead':
            punct = checkpunct(doc)
            if punct == 0:
                prev, ind = checkprev(df, index)
                if prev not in finsent:
                    fullsent = checkprevcond(prev, doc, ind, df)
                    return fullsent
            else:
                return doc.text
        elif tok.text.lower() == 'them' and (tok.dep_ == 'dobj'):
            punct = checkpunct(doc)
            if punct == 0:
                prev, ind = checkprev(df, index)
                fullsent = prev + doc.text
                return fullsent
            else:
                return doc.text
        elif tok.text.lower() == 'so' and (tok.dep_ == 'advmod'):
            prev, ind = checkprev(df, index)
            fullsent = prev + doc.text
            return fullsent
        elif tok.text.lower() == 'so' and (tok.dep_ == 'cc'):
            return doc.text
        elif tok.text.lower() == 'then':
            if doc[0].text == tok.text:
                prev, ind = checkprev(df, index)
                fullsent = checkprevcond(prev, doc, ind, df)
                return fullsent
            else:
                return doc.text
        elif tok.text == 'These':
            if doc[0].text == tok.text:
                prev, ind = checkprev(df, index)
                fullsent = checkprevcond(prev, doc, ind, df)
                return fullsent

        elif tok.text.lower() == 'such' and (tok.dep_ == 'predet'):
            prev, ind = checkprev(df, index)
            fullsent = checkprevcond(prev, doc, ind, df)
            return fullsent
        elif tok.text.lower() == 'such' and (tok.dep_ == 'predet'):
            return doc.text

        elif tok.text.lower() == 'these' and (tok.dep_ == 'det'):
            return doc.text
        elif tok.text.lower().__contains__ ('For example') :
                prev, ind = checkprev(df, index)
                fullsent = checkprevcond(prev,doc,ind,df)
                return fullsent
        elif tok.text.lower().__contains__ ('For example') :
                prev, ind = checkprev(df, index)
                fullsent = checkprevcond(prev,doc,ind,df)
                return fullsent

        elif tok.text.lower() in bcorefs and (tok.dep_ == 'advmod' or tok.dep_ == 'det'):
                prev, ind = checkprev(df, index)
                fullsent = checkprevcond(prev,doc,ind,df)
                return fullsent

def checksimbef(sentss):
    sentis = sentss.split(".")
    length = len(sentis) - 1
    sent = sentis.pop(length)
    l = length -1
    if length > 1:
        for ind, val in enumerate(sentis):
            if ind <l:
                doc_1 = sentis[ind]
                doc_2 = sentis[ind + 1]
                emb1 = model.encode(doc_1, convert_to_tensor=True)
                emb2 = model.encode(doc_2, convert_to_tensor=True)
                score = util.cos_sim(emb1,emb2)[0]
                if score > 0.1:
                    if doc_1 not in simdoc:
                        simdoc.append(doc_1)
                        simdoc.append(doc_2)
                    else:
                        simdoc.append(doc_2)
    else:
        simdoc.append(sentis[0])
    return simdoc

def checksimaft(sentsi):
    sentis = sentsi.split(".")
    length = len(sentis) - 1
    sent = sentis.pop(length)
    l = length -1
    if length > 1:
        for ind, val in enumerate(sentis):
            if ind <l:
                doc_1 = sentis[ind]
                doc_2 = sentis[ind + 1]
                emb1 = model.encode(doc_1, convert_to_tensor=True)
                emb2 = model.encode(doc_2, convert_to_tensor=True)
                score = util.cos_sim(emb1, emb2)[0]
                if score > 0.1:
                    if doc_1 not in simdoc:
                        simdoc.append(doc_1)
                        simdoc.append(doc_2)
                    else:
                        simdoc.append(doc_1)
    else:
        simdoc.append(sentis[0])
    return simdoc

def checknext(dfrm,rows,ind):
    if ind < len(dfrm.index) - 1:
        ind = ind + 1
        nexts = str(dfrm.iloc[ind].item())
        if nexts not in finsent:
            result = [f for f in nextcons if re.search(f,nexts)]
            if result!= []:
                nexti = nexts
            else:
                nexti = None
    else:
        nexti = None
    return nexti

def checkafconst(dfrm,rows,ind):
    if ind < len(dfrm.index)-1:
        ind = ind + 1
        nexts = str(dfrm.iloc[ind].item())
        if nexts not in fsent:
            ac = [a for a in acorefs if a in rows.text]
            nx = [n for n in nextcons if n in nexts]
            if ac or nx != []:
                next = rows.text + nexts
            else:
                next = rows.text
    else:
        next = rows.text
    return next

def checkprev(dfrm,ind):
    ind = ind - 1
    prev = str(dfrm.iloc[ind].item())
    return prev, ind

def checkstags(dfsx,dftx):
    global fullsent
    for ind, row in dfsx.itertuples():
        srow = sent_tokenize(row)
        rcom = [y for y in strgcorefs if y in row]
        skp =  [f for f in skip if re.search(f, row)]
        if skp == []:
            if rcom != []:
                for inds,rows in dftx.itertuples():
                    for p in strong_words:
                        if re.search(p, rows):
                            if rows == srow[0]:
                                pre, index = checkprev(dftx, inds)
                                doc = nlp(row)
                                fullsent = checkprevcond(pre,doc,index,dftx)

                            else:
                                continue
            else:
                fullsent = row
        else:
            continue
        strongtexts.append(fullsent)
    return strongtexts

def checktable(df):
    i = len(df.columns)
    i1 = i-1
    for row in df.itertuples():
        for i2 in range(i1):
            rows = str(row[i2])
            if any(e in rows for e in constraints):
                ctb.append(rows)
    return ctb

def checktext(df):
    global upsent
    df1 = pd.DataFrame()
    for index,rows in df.itertuples():
        if any(x in rows.lower() for x in constraints):
            result = [f for f in skip if re.search(f, rows)]
            if result == []:
                dup = [s for s in fsent if rows in s]
                if dup == []:
                    if any(y in rows.lower() for y in bcorefs):
                        doc = nlp(rows)
                        sents = checkbfconst(df,doc, index)
                        nexts = checknext(df,doc, index)
                        if nexts is not None:
                            senti = sents + nexts
                        else:
                            senti = sents
                        if senti is not None:
                            upsent = checksimbef(senti)

                    else:
                        doc = nlp(rows)
                        nexti = checkafconst(df,doc, index)
                        if nexti is not None:
                            upsent = checksimaft(nexti)
        if len(upsent) > 0:
            finsent.extend(upsent)
            finalsent = '.'.join(upsent)
            fsent.append(finalsent)
        upsent.clear()
    #df1 = df1.append(fsent)
    df1 = pd.DataFrame(fsent)
    return df1

def main():
    dla = input("Enter the choice of input: \n A.Table B.Strong Tags C.Text [A/B/C]?")
    if dla == 'A':
        dft = pd.read_excel("C:\\Users\\priya\\Downloads\\table.xlsx",sheet_name='Sheet')
        result = checktable(dft)
        df1 = pd.DataFrame(result)
        writer = ExcelWriter("C:\\Users\\priya\\Downloads\\table.xlsx",mode = 'a')
        df1.to_excel(writer, 'Sheet1')
        writer.save()

    if dla == 'B':
        dfsx = pd.read_excel("C:\\Users\\priya\\Downloads\\Test files\\Batch_Files1\\constout.xlsx",header=None)
        root = "C:\\Users\\priya\\Downloads\\Test Files"
        for root, dirs, files in os.walk(root):
            for name in files:
                r = os.path.join(root, name)
                if os.path.isfile(r) and name.endswith('.txt'):
                    with open(r, encoding="utf8") as file:
                        for line in file:
                            sentences = sent_tokenize(line)
                            sents.extend(sentences)
                # sentences = line.split(".")
        dftx = pd.DataFrame(sents)
        df1 = checkstags(dfsx,dftx)
        df1 = pd.DataFrame(df1)
        writer = ExcelWriter("C:\\Users\\priya\\Downloads\\strongtags.xlsx")
        df1.to_excel(writer)
        writer.save()

    if dla == 'C':
        with open("C:\\Users\\priya\\Downloads\\test1.txt", encoding="utf8") as file:
            for line in file:
                sentences = sent_tokenize(line)
                # sentences = line.split(".")
            df = pd.DataFrame(sentences)
            df1 = checktext(df)
            # datatoexcel = pd.ExcelWriter("C:\\Users\\priya\\Desktop\\original.xlsx")
   # df.to_excel("C:\\Users\\priya\\Desktop\\original.xlsx")
    #print(df)
            df1.to_excel("C:\\Users\\priya\\Desktop\\output.xlsx")
            print(df1)
    #res = kmeans(df1)
    #print(res)


if __name__ == "__main__":
    main()



