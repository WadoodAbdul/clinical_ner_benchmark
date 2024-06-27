
# Intro 

> A benchmark is a set of tasks/datasets

### Expected Behaviour

when you 
1. load a model(compatible with the task)
2. load a benchmark(task_datasets)
3. and pass them to an evaluation pipeline, 
    you want the predicted output, evaluation results, time and other artifacts

### Functionality
 
Based on the expected behaviour, 
- Each dataset should belong to a task, describing it's expected inputs & outputs.
- For a model to be compatible with a task, it should have a method to execute the task. {input: dataset_input, output: dataset_output}
- To evaluate the model, we need a defined metric. This is a property of the dataset too.



# Conceptual Guides

The core components of the benchmarking process are:-

1. Model
2. Dataset
3. Metric
4. Evaluator

## Model

