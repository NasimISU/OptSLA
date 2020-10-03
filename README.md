<html>
<h1>README</h1>
<h2>OPTSLA: an Optimization-Based Approach for Sequential LabelAggregation </h2>
<p>OPTSLA is an Optimization-based Sequential Label Aggregation method, that jointly considers the characteristics of
 sequential labeling tasks, workers reliabilities, and advanced deep learning techniques to conquer the challenge of annotation aggregation.</p>

<h3>Structure</h3>
The code is divided into 2 folders, code and dataset. The dataset folder contains the
files NER dataset formatted into conll format. Code folder contains python files.

<h4>Embedding</h4>

We use Glove for embedding, please download glove.6B from below link, unzip and place unzipped
files in OPTSLA folder.

http://nlp.stanford.edu/data/glove.6B.zip

<h4>Python file details</h4>

 <table style="width:100%">
  <tr>
    <th>Python file</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>conlleval.py</td>
    <td>This file is used for evaluating the output</td>
  </tr>
  <tr>
    <td>data_preprocessing.py</td>
    <td>This file is used for data preprocessing</td>
  </tr>
  <tr>
    <td>evaluation.py</td>
    <td>This is the main file containing OPTSLA implementation</td>
  </tr>
  <tr>
    <td>functions.py</td>
    <td>This file contains implementation of few necessary functions</td>
  </tr>
</table> 

<h3>Usage</h3>
The following is the sequence of execution:

NOTE: Please update variables in python files before proceeding.

Pre-processing of the data is the first step, to perform the task execute below command
<ul>
  <li>python data_preprocessing.py</li>
</ul> 

A new folder named iteration0 is created in execution folder which contains pre-processed files.

Now, execute OPTSLA by running following command
<ul>
  <li>python evaluation.py</li>
</ul> 

Once aggregation is done, the model can be evaluated by running following command
<ul>
  <li>python calculations.py</li>
</ul>

This will create a file in calculations folder with the results.


<h3>Contact</h3>

In case of any queries, please contact us at
<ul>
  <li>aditkulk@iastate.edu</li>
  <li>nasim@iastate.edu</li>
</ul>

<h3>Evaluation Note</h3>
The dataset provided contains 4515 sentences, the results published in the paper are evaluated on 3466 sentences to match with baselines.

<h3>References</h3>
https://arxiv.org/abs/1709.01779

</html>