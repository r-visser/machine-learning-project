Preprocessing steps on dataset: 

  - Publishers are extracted from the url's such that we can group them (useful for splitting them in training and test). 
  - Most common sentences (at least 2 words) that occured at least 50 times removed. Because we think these are advertisements or footnotes on the scraped sites. 
  - White space characters removed ('\n', '....')
  - 'Advertisement' removed 
  - Only kept articles with amount_of_tokens > 400 && amount_of_tokens < 3000. An A4 page can have 3000 characters. 
  - All articles by a certain publisher can only exist in the training OR test set. 