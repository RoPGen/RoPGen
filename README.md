## RoPGen: Towards Robust Code Authorship Attribution via Automatic Coding Style Transformation

We propose an innovative framework dubbed Robust coding style Patterns Generation (RoPGen), which essentially learns authors’ unique coding style patterns that are hard for attackers to manipulate or imitate. The key idea is to incorporate data augmentation and gradient augmentation to learn robust coding style patterns. 

GitHub-C and GCJ-Java datasets are created by ourselves. 

(1). GitHub-C dataset. 
 
 We create this dataset from GitHub, by crawling the C programs of authors who contributed between 11/2020 and 12/2020. We filter the repositories that are marked as forks (because they are duplicates) and the repositories that simply duplicate the files of others. We preprocess these files by removing the comments; we then eliminate the resulting files that (i) contain less than 30 lines of code because of their limited functionalities or (ii) overlap more than 60% of its lines of code with other files.
 
(2). GCJ-Java dataset. 

We create this dataset from GCJ between 2015 and 2017. Since some authors participate in GCJ for multiple years, we merge their files according to their IDs. We select the
authors who have written at least 30 Java program files. The dataset has 2,399 Java files of 74 authors, with an average of 135 lines of code per file.
