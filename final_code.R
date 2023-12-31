#任务一：从中任选三个数据集，选一种数据挖掘算法通过训练三种不同数据集进行比较分析
#选用三种不同的数据挖掘算法，完成违约判别的分类预测（至少用三种数据挖掘算法实现，如支持向量机、神经网络、决策树等），并进行不同算法优劣的深入比较分析。

#主体如下
#导入数据
install.packages("openxlsx")
library('openxlsx')
library('readxl')
Aus<-read.xlsx("D:/R语言/脚本/data/Australiadata.xlsx",sheet='Sheet1',startRow=1,colNames=TRUE)
Jap<-read.xlsx("D:/R语言/脚本/data/Japandata.xlsx",sheet='Sheet1',startRow=1,colNames=TRUE)
Tai<-read_excel("D:/R语言/脚本/data/Taiwan.xls")

#进行数值化标准化处理
Aus$Y<- as.factor(Aus$Y)
Jap$Y<- as.factor(Jap$Y)
Tai$Y<- as.factor(Tai$Y)
#拆分训练集和测试集
index_Aus <- sample(2,nrow(Aus),replace = TRUE,prob=c(0.7,0.3))
traindata_Aus <- Aus[index_Aus==1,]
testdata_Aus <- Aus[index_Aus==2,]
index_Jap <- sample(2,nrow(Jap),replace = TRUE,prob=c(0.7,0.3))
traindata_Jap <- Jap[index_Jap==1,]
testdata_Jap <- Jap[index_Jap==2,]
index_Tai <- sample(2,nrow(Tai),replace = TRUE,prob=c(0.7,0.3))
traindata_Tai <- Tai[index_Tai==1,]
testdata_Tai <- Tai[index_Tai==2,]
index_Tai <- sample(2,nrow(Tai),replace = TRUE,prob=c(0.7,0.3))
Tai_1 <- Tai[index_Tai==1,]
Tai_2 <- Tai[index_Tai==2,]
Tai_3 <- Tai[index_Tai==3,]
#随机森林处理
library("randomForest")
set.seed(12345)
#先将tree设为800进行测试（400-600之间稳定）
rFM1<-randomForest(Y~.,data=Aus,ntree=800,importance=TRUE,proximity=TRUE)
plot(rFM1)


rFM1<-randomForest(Y~.,data=traindata_Aus,ntree=450,importance=TRUE,proximity=TRUE)
rFM2<-randomForest(Y~.,data=Jap,ntree=450,importance=TRUE,proximity=TRUE)
rFM3<-randomForest(Y~.,data=Tai,ntree=450,importance=TRUE,proximity=TRUE)
head(rFM1$votes)#观测的各类别预测概率
head(rFM2$votes)#观测的各类别预测概率
head(rFM3$votes)#观测的各类别预测概率
head(rFM1$oob.times)#各观测作为袋外观测的次数
head(rFM2$oob.times)#各观测作为袋外观测的次数
head(rFM3$oob.times)#各观测作为袋外观测的次数
#预测
Aus_pred<-predict(rFM1, newdata=testdata_Aus)
Aus_matrix<-table(Aus_pred,testdata_Aus$Y)#混淆矩阵
Jap_pred<-predict(rFM2, newdata=testdata_Jap)
Jap_matrix<-table(Jap_pred,testdata_Jap$Y)#混淆矩阵
Tai_pred<-predict(rFM3, newdata=testdata_Tai)
Tai_matrix<-table(Aus_pred,testdata_Jap$Y)#混淆矩阵
#错判率输出
er_Aus<-(sum(Aus_matrix)-sum(diag(Aus_matrix)))/sum(Aus_matrix)
er_Jap<-(sum(Jap_matrix)-sum(diag(Jap_matrix)))/sum(Jap_matrix)
er_Tai<-(sum(Tai_matrix)-sum(diag(Tai_matrix)))/sum(Tai_matrix)
#绘制图像（有问题）
install.packages("pROC")
library("pROC")
roc1<-roc(as.ordered(testdata_Aus$Y) ,as.ordered(Aus_pred))
plot(roc1,print.auc=T, auc.polygon=T, grid=c(0.1, 0.2), grid.col=c("green","red"),max.auc.polygon=T, auc.polygon.col="skyblue",print.thres=T)


#不同方法
#神经网络
library('neuralnet')
set.seed(12345)
BPnet1<-neuralnet(Y~.,data=traindata_Tai,hidden=2,err.fct="sse",linear.output=FALSE)
BPnet1$result.matrix #连接权重以及其他信息
BPnet1$weights #连接权重列表
plot(BPnet1)
Tai_out<-compute(BPnet1,covariate=test)
Tai_out$net.result
pred<-prediction(predictions=as.vector(BPnet1$net.result),labels=BPnet1$response)
par(mfrow=c(2,1))
perf<-performance(pred,measure="tpr",x.measure="fpr")#计算标准ROC曲线
plot(perf,colorsize=TRUE,print.cutoffs.at)
perf<-performance(pred,measure="acc")
Tai_pred<-cbind(BPnet1$resonse,BPnet1$net.result[[1]])
Tai_pred<-cbind(Tai_pred,ifelse(Tai_pred[,2]>0.3,1,0))
confM_BP<-table(Tai_pred[,1],Tai_pred[,3])
Tai_matrix<-table(Tai_pred,testdata_Tai$Y)#混淆矩阵
er_Tai<-(sum(Tai_matrix)-sum(diag(Tai_matrix)))/sum(Tai_matrix)
library("pROC")
roc1<-roc(as.ordered(testdata_Tai$Y) ,as.ordered(Tai_pred))
plot(roc1,print.auc=T, auc.polygon=T, grid=c(0.1, 0.2), grid.col=c("green","red"),max.auc.polygon=T, auc.polygon.col="skyblue",print.thres=T)


#支持向量机

install.packages('e1071')
set.seed(12345)
tobj<-tune.svm(Y~.,data=Tai_train,type="C-classification",kernel="radial",cost=10^(-6:-3),gamma=10^(-3:2))
BestSvm<-tobj$best.model
summary(BestSvm)
yPred<-predict(BestSvm,Tai_test)
confM<-table(yPred,Tai_test$Y)
confM
err<-(sum(confM)-sum(diag(confM)))/sum(confM)

pred1<-predict(rFM1, newdata=test)
matrix1<-table(pred1,test$Y)#混淆矩阵
err1<-(sum(matrix1)-sum(diag(matrix1)))/sum(matrix1)
w1<-(0.178499-err1)/450