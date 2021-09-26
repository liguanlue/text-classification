# text-classification
Text Classification Based on Naive Bayes 

1. Data Processing

	1.1 Data preprocessing
	
		(1) Word segmentation: Use the pseg.cut method in the jieba library to get the result of word segmentation and the part of speech of each word
		(2) Delete designated parts of speech: such as adjectives, adverbs, conjunctions, pronouns, prepositions, time, etc.
		(3) Delete stop words
		(4) Divide the original data into training set and test set, and generate corresponding label files
		
	1.2 Feature extraction
	
		(1) Calculate word frequency
		(2) Represent the document as a word vector
		(3) The sparse matrix represented by the output word vector
	
	1.3 tf-idf
	
		(1) Express the document as the value of tf-idf in the form of a sparse matrix
		(2) Use the document represented by tf-idf as the training data of the Bayesian classification system
		
	
