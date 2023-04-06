import sys
fin=sys.argv[1]
def convert_to_trec(fin):
    with open(fin) as f:
        lines=f.readlines()
    saved=[]
    for line in lines:
        qid,pid,index,score=line.strip().split()
        saved.append(str(int(float(qid)))+'\t'+"vanilla_bert\t"+str(int(float(pid)))+'\t'+str(index)+'\t'+str(score)+'\tvanilla_bert\n')
    with open(fin+".trec","w") as f:
        f.writelines(saved)

if __name__=="__main__":
    convert_to_trec(fin)