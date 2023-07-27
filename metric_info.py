from datasets import load_metric
metrics_list = list_metrics()
l = len(metric_list)
for x in range(0,l):
    metric = load_metric(metrics_list[1])
    print(metric.inputs_description)
