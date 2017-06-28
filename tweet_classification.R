rm(list=ls())
setwd("D:/rakesh/data science/Project/tweet/TweetsClassification")

#load libraries
library(stringr)
library(tm)
library(wordcloud2)
library(slam)
library(sentiment)

#Load comments/text
tweet_data = read.csv("TweetsDataSet.csv", header = T)
str(tweet_data)

#Load defined stop words
#stop_words = read.csv("stopwords.csv", header = T)
#names(stop_words) = "StopWords"

#Delete the leading spaces
tweet_data$tweet = str_trim(tweet_data$tweet)

#Select only text column
Tweet = data.frame(tweet_data[,2])
names(Tweet) = "comments"
Tweet$comments = as.character(Tweet$comments)

##Pre-processing
#convert comments into corpus
TweetCorpus = Corpus(VectorSource(Tweet$comments))
writeLines(as.character(TweetCorpus[[2]]))
#TweetCorpus = tm_map(TweetCorpus, PlainTextDocument)

#case folding
TweetCorpus = tm_map(TweetCorpus, tolower)

#remove stop words
TweetCorpus = tm_map(TweetCorpus, removeWords, stopwords('english'))

#remove punctuation marks
TweetCorpus = tm_map(TweetCorpus, removePunctuation)

#remove num bers
TweetCorpus = tm_map(TweetCorpus, removeNumbers)

#remove unnecesary spaces
TweetCorpus = tm_map(TweetCorpus, stripWhitespace)

#convert into plain text
TweetCorpus = tm_map(TweetCorpus, PlainTextDocument)

#create corpus
TweetCorpus = Corpus(VectorSource(TweetCorpus))

##wordcloud
#Remove the defined stop words
TweetCorpus_WC = TweetCorpus
TweetCorpus_WC = tm_map(TweetCorpus, removeWords, c('i','its','it','us','use','used','using','will','yes','say','can','take','one',
                                                  stopwords('english')))
#TweetCorpus_WC = tm_map(TweetCorpus_WC, removeWords, stop_words$StopWords)

#Word cloud
wordcloud(TweetCorpus_WC, max.words = 100, scale=c(3, .1), 
          random.order = F, colors=brewer.pal(6, "Dark2"))

#Another method to build wordcloud
pal2 = brewer.pal(8,"Dark2")
png("wordcloud1.png", width = 12, height = 8, units = 'in', res = 300)
wordcloud(TweetCorpus_WC, scale = c(5,.2), min.freq = 30, max.words = 150, random.order = FALSE, rot.per = .15, colors = pal2)
dev.off()

#Build document term matrix
tdm = TermDocumentMatrix(TweetCorpus)
#tdm_min = TermDocumentMatrix(TweetCorpus, control=list(weighting=weightTfIdf, minWordLength=4, minDocFreq=10))

#calculate the terms frequency
words_freq = rollup(tdm, 2, na.rm=TRUE, FUN = sum)
words_freq = as.matrix(words_freq)
words_freq = data.frame(words_freq)
words_freq$words = row.names(words_freq)
row.names(words_freq) = NULL
words_freq = words_freq[,c(2,1)]
names(words_freq) = c("Words", "Frequency")
#wordcloud(words_freq$Words, words_freq$Frequency)

#Most frequent terms which appears in atleast 700 times
findFreqTerms(tdm, 100)

#sentiment Analysis
#Another method
library(RSentiment)
Sys.getenv("JAVA_HOME")
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_121') # for 64-bit version
#Sys.setenv(JAVA_HOME='C:\\Program Files (x86)\\Java\\jre1.8.0_121') # for 32-bit version
library(rJava)
df = calculate_sentiment(Tweet$comments)

#Another method
#Install sentiment library using binay source
#install.packages("D:/sentiment_0.2.tar.gz", repos = NULL, type="source")
library(sentiment)

#classifying the corpus as negative and positive and neutral
polarity = classify_polarity(Tweet$comments, algorithm = "bayes", verbose = TRUE)
polarity = data.frame(polarity)

#Attached sentiments to the comments
newdocs = cbind(Tweet, polarity)

#separate the comments based on polarity
newdocs_positive = newdocs[which(newdocs$BEST_FIT == "positive"),]
newdocs_negative = newdocs[which(newdocs$BEST_FIT == "negative"),]
newdocs_neutral  = newdocs[which(newdocs$BEST_FIT == "neutral"),]  

##Build word cloud for each  polarity
#Positive terms wordcloud
#Pre-processing
posCorpus = Corpus(VectorSource(newdocs_positive$comments))
posCorpus = tm_map(posCorpus, tolower)
posCorpus = tm_map(posCorpus, removeWords, stopwords('english'))
posCorpus = tm_map(posCorpus, removePunctuation)
posCorpus = tm_map(posCorpus, removeNumbers)
posCorpus = tm_map(posCorpus, stripWhitespace)
posCorpus = tm_map(posCorpus, PlainTextDocument)
posCorpus = Corpus(VectorSource(posCorpus))

#Remove the defined stop words
posCorpus = tm_map(posCorpus, removeWords, c('i','its','it','us','use','used','using','will','can',
                                             stopwords('english')))
#posCorpus = tm_map(posCorpus, removeWords, stop_words$StopWords)

#wordcloud(posCorpus, max.words = 200, scale=c(3, .1), colors=brewer.pal(6, "Dark2"))

# wordcloud
pal2 = brewer.pal(8,"Dark2")
png("wordcloud_posv1.png", width=12,height=8, units='in', res=300)
wordcloud(posCorpus, scale=c(5,.2),min.freq=30, max.words=150, random.order=FALSE, rot.per=.15, colors=pal2)
dev.off()

#Wordcloud for negative terms
negCorpus = Corpus(VectorSource(newdocs_negative$comments))
negCorpus = tm_map(negCorpus, tolower)
negCorpus = tm_map(negCorpus, removeWords, c('will','can','thanks','better','get','also',
                                             'well','good','now', stopwords('english')))
negCorpus = tm_map(negCorpus, removePunctuation)
negCorpus = tm_map(negCorpus, removeNumbers)
negCorpus = tm_map(negCorpus, stripWhitespace)
negCorpus = tm_map(negCorpus, PlainTextDocument)
negCorpus = Corpus(VectorSource(negCorpus))

#wordcloud(negCorpus, max.words = 200, scale=c(3, .1), colors=brewer.pal(6, "Dark2"))

# wordcloud
pal2 = brewer.pal(8,"Dark2")
png("wordcloud_negv1.png", width=12,height=8, units='in', res=300)
wordcloud(negCorpus, scale=c(5,.2),min.freq=20, max.words=150, random.order=FALSE, rot.per=.15, colors=pal2)
dev.off()

