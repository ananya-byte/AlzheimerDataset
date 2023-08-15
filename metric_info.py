from datasets import load_metric
import evaluate


def tp_tn_1(comp1):
    tp1 = comp1[0][0]
    fp1 = comp1[0][1]+comp1[0][2]+comp1[0][3]
    fn1 = comp1[1][0]+ comp1[2][0]+comp1[3][0]
    tn1 = comp1[1][1]+ comp1[2][2]+comp1[3][3]

