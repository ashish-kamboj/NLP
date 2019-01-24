###########################################################################################################################
###                                 :: Online Patient conversation classification ::
###########################################################################################################################

### Loading Libraries
  library(tm)
  library(wordcloud)
  library(caret)
  library(kernlab)

### Loading Data
  df_train <- read.csv("train.csv", stringsAsFactors = FALSE, na.strings=c("","NA"))
  df_test <- read.csv("test.csv", stringsAsFactors = FALSE, na.strings=c("","NA"))
  
  
### Combining train and test datasets

  ## Adding Index column in train dataset(as it is present in test dataset)
    df_train$Index <- c(572:1728)
    df_train <- df_train[,c(10,1:9)] #Re-arranging the test dataset columns
    
  ## Adding "Patient_Tag" column in test dataset
    df_test$Patient_Tag <- NA
    df_test <- subset(df_test, select = -c(X))
    
    df_all <- rbind(df_train, df_test) #Combined train and test datasets
    
### Data Cleaning
  ## Correcting Source Name
    df_all$Source <- replace(df_all$Source,df_all$Source=="Facebook","FACEBOOK")
    
  ## Cleaning the "TRANS_CONV_TEXT" column
    trans_conv_text <- VCorpus(VectorSource(df_all$TRANS_CONV_TEXT))
    trans_conv_text <- tm_map(trans_conv_text, content_transformer(stripWhitespace))
    trans_conv_text <- tm_map(trans_conv_text, content_transformer(tolower))
    trans_conv_text <- tm_map(trans_conv_text, content_transformer(removeNumbers))
    trans_conv_text <- tm_map(trans_conv_text, content_transformer(removePunctuation))
    trans_conv_text <- tm_map(trans_conv_text, removeWords, stopwords("english"))
    trans_conv_text <- tm_map(trans_conv_text, stemDocument,language = "english") #perform stemming 
    
    # Creating document term matrices for the trans_conv_text
      trans_conv_text.dtm <- DocumentTermMatrix(trans_conv_text, control=list(wordLengths=c(1,Inf)))
      dim(trans_conv_text.dtm)
      
    # Let's remove the variables which are 95% or more sparse
      new_trans_conv_text.dtm <- removeSparseTerms(trans_conv_text.dtm,sparse = 0.95)
      dim(new_trans_conv_text.dtm)
      
    # Calculate the most frequent and least frequently occurring set of features
      #Find frequent terms
        colS <- colSums(as.matrix(new_trans_conv_text.dtm))
        length(colS)
        doc_features <- data.table(name = attributes(colS)$names, count = colS)
      
    # Most frequent and least frequent words
      doc_features[order(-count)][1:10] #top 10 most frequent words
      doc_features[order(count)][1:10] #least 10 frequent words
      
    # Create wordcloud
      wordcloud(names(colS), colS, min.freq = 100, scale = c(6,.1), 
                colors = brewer.pal(6, 'Dark2'))
      
      
    # Creating a dataframe based on the "trans_conv_text.dtm" column names
      df_trans_conv_text <- as.data.frame(as.matrix(new_trans_conv_text.dtm))  
      
  ## Cleaning the "Title" column
    title <- VCorpus(VectorSource(df_all$Title))
    title <- tm_map(title, content_transformer(stripWhitespace))
    title <- tm_map(title, content_transformer(tolower))
    title <- tm_map(title, content_transformer(removeNumbers))
    title <- tm_map(title, content_transformer(removePunctuation))
    title <- tm_map(title, removeWords, stopwords("english"))
    title <- tm_map(title, stemDocument,language = "english") #perform stemming 
    
    # Creating document term matrices for the trans_conv_text
      title.dtm <- DocumentTermMatrix(title, control=list(wordLengths=c(1,Inf)))
      dim(title.dtm)
    
    # Let's remove the variables which are 95% or more sparse
      new_title.dtm <- removeSparseTerms(title.dtm,sparse = 0.95)
      dim(new_title.dtm)
      
    # Creating a dataframe based on the "trans_conv_text.dtm" column names
      df_title<- as.data.frame(as.matrix(new_title.dtm)) 
      

  ## One-hot encoding for the "Source" column
    dmy <- dummyVars("~ Source", data = df_all, fullRank = T)
    dummy_1 <- data.frame(predict(dmy,newdata = df_all))
    
    
### Retrieving the correct labels
  df <- cbind(Index=df_all$Index, df_title, df_trans_conv_text, Patient_Tag=df_all$Patient_Tag)

    
### Splitting into train and test dataset
  train <- df[!df$Index %in% c(1:571),]
  test <- df[df$Index %in% c(1:571),]
  
  train <- train[,-1] #Removing the "Index" column
  test <- test[,-457] #Removing the "Patient_Tag" column
  
  ## Changing output variable "Patient_Tag" to factor type 
    train$Patient_Tag <- as.factor(train$Patient_Tag)
  
    
### Model Building(base Model)
  ksvm.model1 <- ksvm(Patient_Tag~., data= train, kernel="rbfdot")
  
  
### Hyperparameter tuning and Cross Validation 
  
  trainControl <- trainControl(method="cv", number=5)
  metric <- "Accuracy" 
  
  ## Performing 5-fold cross validation "svmRadial"
  set.seed(50)
  grid_radial <- expand.grid(.sigma=seq(1.50, 3.00, by=0.20), .C=seq(1, 5, by=1))  # making a grid of C and sigma values. 
  fit.svm_radial <- train(Patient_Tag~., data=train, method="svmRadial", metric=metric, 
                          tuneGrid=grid_radial, trControl=trainControl)
  
  print(fit.svm_radial) # The final values used for the model were sigma = 2.9 and C = 1.
  plot(fit.svm_radial)
  
  
### Final Model
  ksvm.model2 <- ksvm(Patient_Tag~., data= train, kernel="rbfdot", sigma=2.9, C=1)
  

### Prediction on Test Data
  pred <- predict(ksvm.model2, test)
  
  sample_submission <- data.frame(Index =test$Index, Patient_Tag = pred)
  write.csv(sample_submission, "submission.csv", row.names = FALSE)
  
  
 
