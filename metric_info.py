from datasets import list_metrics,load_metric
metrics_list = list_metrics()
l = len(metrics_list)
for x in range(0,l):
    metric = load_metric(metrics_list[l])
    print(metric.inputs_description)
