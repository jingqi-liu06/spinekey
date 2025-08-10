install.packages("jsonlite")
install.packages("ggpubr")
library(ggpubr)
library(jsonlite)
library(ggplot2)
library(pROC)
library(dplyr)

json_file<-"../data/full/output/eval_results.json"
json_data<-fromJSON(paste(readLines(json_file),collaspse=""))

out1 <- data.frame(matrix(double(),17,5))
colnames(out1) <- c("sort","Mean","Var","CI","IR")
out1$sort <- c("T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","L1","L2","L3","L4","L5")
out2 <- data.frame(matrix(double(),17,5))
colnames(out2) <- c("sort","Mean","Var","CI","IR")
out2$sort <- c("T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","L1","L2","L3","L4","L5")
#建立输出集1\2

df1 <- data.frame(matrix(numeric(),800,17))
colnames(df1) <- c("T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","L1","L2","L3","L4","L5")
for(k in 8:24){
  for(i in 1:800){
    df1[[k-7]][[i]] <- (json_data[[2]][[i]][[2]][[k]]+json_data[[2]][[i]][[3]][[k]])/(json_data[[2]][[i]][[2]][[7]]+json_data[[2]][[i]][[3]][[7]])
    }
}
#计算各除以C7的值

out1$Mean <- c(0)
for(i in 1:17){
  for(k in 1:800){
    out1$Mean[[i]] <- df1[[i]][[k]]+out1$Mean[[i]]
  }
  out1$Mean[[i]] <- out1$Mean[[i]]/800
}
#计算平均值

for(i in 1:16){
  out1$IR[i] <- out1$Mean[i+1]-out1$Mean[i]
  out1$Var[[i]] <- var(df1[[i]])
}
for(i in 1:17){
  out1$Var[[i]] <- var(df1[[i]])
}
#计算斜率及方差

shapiro.test(df1$T1)
#正态性检验
wilcox.test(df1$T1,df1$T2)
#wilcox检验

df2 <- data.frame(matrix(numeric(),800,17))
colnames(df2) <- c("T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","L1","L2","L3","L4","L5")
for(k in 8:24){
  for(i in 1:800){
    df2[[k-7]][[i]] <- (json_data[[2]][[i]][[2]][[k]]+json_data[[2]][[i]][[3]][[k]])/(json_data[[2]][[i]][[2]][[k-1]]+json_data[[2]][[i]][[3]][[k-1]])
  }
}

out2$Mean <- c(0)
for (i in 1:17) {
  for (k in 1:800){
    out2$Mean[[i]] <- df2[[i]][[k]]+out2$Mean[[i]]
  }
  out2$Mean[[i]] <- out2$Mean[[i]]/800
}

for (i in 1:16) {
  out2$IR[[i]] <- out2$Mean[[i+1]]-out2$Mean[[i]]
}
for(i in 1:17){
  out2$Var[[i]] <- var(df2[[i]])
}

ggplot(df1,aes(T1))+
         geom_histogram(bins = 50, fill = 4) +
         labs(x = "Normal T1/C7")

for(i in 1:17){
  out1$CI[[i]] <- t.test(df1[i], alternative = "greater")$conf.int[[1]]
  out2$CI[[i]] <- t.test(df2[i], alternative = "greater")$conf.int[[1]]
}
#计算95%置信区间的左界

write.csv(out2,file = "out2.csv",row.names = FALSE)

setwd("C:/Users/11215/Desktop/算死我吧/data2")
wsf <- read.csv("wholespine.csv")
psf <- read.csv("partlyspine.csv")
sws <- read.csv("suswholespine.csv")
sps <- read.csv("suspartlyspine.csv")
TW<-"C:/Users/11215/Desktop/算死我吧/data/胸椎压缩性骨折全脊柱.json"
data_TW<-fromJSON(paste(readLines(TW),collaspse=""))
TP<-"C:/Users/11215/Desktop/算死我吧/data/胸椎压缩性骨折局部.json"
data_TP<-fromJSON(paste(readLines(TP),collaspse=""))
CW<-"C:/Users/11215/Desktop/算死我吧/data/腰椎压缩性骨折全脊柱.json"
data_CW<-fromJSON(paste(readLines(CW),collaspse=""))
CP<-"C:/Users/11215/Desktop/算死我吧/data/腰椎压缩性骨折局部.json"
data_CP<-fromJSON(paste(readLines(CP),collaspse=""))
#导入数据

outsf <- data.frame(matrix(double(0),17,5))
colnames(outsf) <- c("sort","Mean","Var","CI","IR")
outsf$sort <- c("T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","L1","L2","L3","L4","L5")
#建立输出集

dfsf <- data.frame(matrix(numeric(),500,17))
colnames(dfsf) <- c("T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","L1","L2","L3","L4","L5")
dfsf[1,] <- 1
#建立数据集

for(i in 2:1455){
  if(psf$Filename[[i]] == psf$Filename[[i-1]] && psf$Filename[[i]] == psf$Filename[[i+1]]){
    for(k in 1:17){
      if(psf$Quad.Label[[i]] == colnames(dfsf[k])){
        dfsf[[k]][[1]] <- dfsf[[k]][[1]]+1
        dfsf[[k]][[dfsf[[k]][[1]]]]  <- (psf[[3]][[i]]+psf[[4]][[i]])/(psf[[3]][[i-1]]+psf[[4]][[i-1]])
      }  
    }
  }
  if(psf$Filename[[i]] == psf$Filename[[i-1]] && psf$Filename[[i]] != psf$Filename[[i+1]] && psf$Filename[[i-1]]!=psf$Filename[[i-2]] && psf$Quad.Label[[i]] == "L5"){
    dfsf[[17]][[1]] <- dfsf[[17]][[1]]+1
    dfsf[[17]][[dfsf[[k]][[1]]]] <- (psf[[3]][[i]]+psf[[4]][[i]])/(psf[[3]][[i-1]]+psf[[4]][[i-1]])
  }
}

for(i in 2:193){
  if(wsf$Filename[[i]] == wsf$Filename[[i-1]] && wsf$Filename[[i]] == wsf$Filename[[i+1]]){
    for(k in 1:17){
      if(wsf$Quad.Label[[i]] == colnames(dfsf[k])){
        dfsf[[k]][[1]] <- dfsf[[k]][[1]]+1
        dfsf[[k]][[dfsf[[k]][[1]]]]  <- (wsf[[3]][[i]]+wsf[[4]][[i]])/(wsf[[3]][[i-1]]+wsf[[4]][[i-1]])
      }  
    }
  }
  if(wsf$Filename[[i]] == wsf$Filename[[i-1]] && wsf$Filename[[i]] != wsf$Filename[[i+1]] && wsf$Filename[[i-1]]!=wsf$Filename[[i-2]] && wsf$Quad.Label[[i]] == "L5"){
    dfsf[[17]][[1]] <- dfsf[[17]][[1]]+1
    dfsf[[17]][[dfsf[[k]][[1]]]] <- (wsf[[3]][[i]]+wsf[[4]][[i]])/(wsf[[3]][[i-1]]+wsf[[4]][[i-1]])
  }
}

for (i in 1:length(data_TW[[1]])) {
  if (length(data_TW[[2]][[i]][[1]]) == 3)
    x[[i]] <- TRUE
  if (length(data_TW[[2]][[i]][[1]]) != 3)
    x[[i]] <- FALSE
}

#检验是否为3脊柱节段

for (i in 1:length(data_TW[[1]])) {
  if(x[[i]] == T){
    for (k in 1:17) {
      if(data_TW[[2]][[i]][[1]][[2]] == colnames(dfsf[k])){
        dfsf[[k]][[1]] <- dfsf[[k]][[1]]+1
        dfsf[[k]][[dfsf[[k]][[1]]]] <- (data_TW[[2]][[i]][[2]][[2]]+data_TW[[2]][[i]][[3]][[2]])/(data_TW[[2]][[i]][[2]][[1]]+data_TW[[2]][[i]][[3]][[1]])
      }
    }
  }
}

for (i in 1:length(data_TP[[1]])) {
  if (length(data_TP[[2]][[i]][[1]]) == 3)
    x[[i]] <- TRUE
  if (length(data_TP[[2]][[i]][[1]]) != 3)
    x[[i]] <- FALSE
}

for (i in 1:length(data_TP[[1]])) {
  if(x[[i]] == T){
    for (k in 1:17) {
      if(data_TP[[2]][[i]][[1]][[2]] == colnames(dfsf[k])){
        dfsf[[k]][[1]] <- dfsf[[k]][[1]]+1
        dfsf[[k]][[dfsf[[k]][[1]]]] <- (data_TP[[2]][[i]][[2]][[2]]+data_TP[[2]][[i]][[3]][[2]])/(data_TP[[2]][[i]][[2]][[1]]+data_TP[[2]][[i]][[3]][[1]])
      }
    }
  }
}

for (i in 1:length(data_CW[[1]])) {
  if (length(data_CW[[2]][[i]][[1]]) == 3)
    x[[i]] <- TRUE
  if (length(data_CW[[2]][[i]][[1]]) != 3)
    x[[i]] <- FALSE
}

for (i in 1:length(data_CW[[1]])) {
  if(x[[i]] == T){
    for (k in 1:17) {
      if(data_CW[[2]][[i]][[1]][[2]] == colnames(dfsf[k])){
        dfsf[[k]][[1]] <- dfsf[[k]][[1]]+1
        dfsf[[k]][[dfsf[[k]][[1]]]] <- (data_CW[[2]][[i]][[2]][[2]]+data_CW[[2]][[i]][[3]][[2]])/(data_CW[[2]][[i]][[2]][[1]]+data_CW[[2]][[i]][[3]][[1]])
      }
    }
  }
}

for (i in 1:length(data_CP[[1]])) {
  if (length(data_CP[[2]][[i]][[1]]) == 3)
    x[[i]] <- TRUE
  if (length(data_CP[[2]][[i]][[1]]) != 3)
    x[[i]] <- FALSE
}

for (i in 1:length(data_CP[[1]])) {
  if(x[[i]] == T){
    for (k in 1:17) {
      if(data_CP[[2]][[i]][[1]][[2]] == colnames(dfsf[k])){
        dfsf[[k]][[1]] <- dfsf[[k]][[1]]+1
        dfsf[[k]][[dfsf[[k]][[1]]]] <- (data_CP[[2]][[i]][[2]][[2]]+data_CP[[2]][[i]][[3]][[2]])/(data_CP[[2]][[i]][[2]][[1]]+data_CP[[2]][[i]][[3]][[1]])
      }
    }
  }
}


for(i in 1:17){
  outsf$Mean[[i]] <- (sum(dfsf[i],na.rm = TRUE)-dfsf[[i]][[1]])/(dfsf[[i]][[1]]-1)
  outsf$Var[[i]] <- var(dfsf[[i]][-1],na.rm = TRUE)
}
#计算平均值、方差

dfsf_subset <- dfsf[2:nrow(dfsf),]
p<-0
for(i in 6:17){
  if(shapiro.test(dfsf_subset[[i]])$p.value[[1]] <= 0.05){
    p[i] <- i
  }
}
p
#结果:L1/L2不正态，其他均正态

for(i in 6:17){
  outsf$CI[[i]] <- t.test(dfsf_subset[i], alternative = "less")$conf.int[[2]]
}

outsf <- outsf[-(1:10), ]
write.csv(outsf,file = "outsf.csv",row.names = F)

outssf <- data.frame(matrix(double(0),17,5))
colnames(outssf) <- c("sort","Mean","Var","CI1","CI2")
outssf$sort <- c("T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","L1","L2","L3","L4","L5")
#建立输出集

dfssf <- data.frame(matrix(numeric(),500,17))
colnames(dfssf) <- c("T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","L1","L2","L3","L4","L5")
dfssf[1,] <- 1
#建立数据集

for(i in 2:843){
  if(sps$Filename[[i]] == sps$Filename[[i-1]] && sps$Filename[[i]] == sps$Filename[[i+1]]){
    for(k in 1:17){
      if(sps$Quad.Label[[i]] == colnames(dfssf[k])){
        dfssf[[k]][[1]] <- dfssf[[k]][[1]]+1
        dfssf[[k]][[dfssf[[k]][[1]]]]  <- (sps[[3]][[i]]+sps[[4]][[i]])/(sps[[3]][[i-1]]+sps[[4]][[i-1]])
      }  
    }
  }
  if(sps$Filename[[i]] == sps$Filename[[i-1]] && sps$Filename[[i]] != sps$Filename[[i+1]] && sps$Filename[[i-1]]!=sps$Filename[[i-2]] && sps$Quad.Label[[i]] == "L5"){
    dfssf[[17]][[1]] <- dfssf[[17]][[1]]+1
    dfssf[[17]][[dfssf[[k]][[1]]]] <- (sps[[3]][[i]]+sps[[4]][[i]])/(sps[[3]][[i-1]]+sps[[4]][[i-1]])
  }
}

for(i in 2:187){
  if(sws$Filename[[i]] == sws$Filename[[i-1]] && sws$Filename[[i]] == sws$Filename[[i+1]]){
    for(k in 1:17){
      if(sws$Quad.Label[[i]] == colnames(dfssf[k])){
        dfssf[[k]][[1]] <- dfssf[[k]][[1]]+1
        dfssf[[k]][[dfssf[[k]][[1]]]]  <- (sws[[3]][[i]]+sws[[4]][[i]])/(sws[[3]][[i-1]]+sws[[4]][[i-1]])
      }  
    }
  }
  if(sws$Filename[[i]] == sws$Filename[[i-1]] && sws$Filename[[i]] != sws$Filename[[i+1]] && sws$Filename[[i-1]]!=sws$Filename[[i-2]] && sws$Quad.Label[[i]] == "L5"){
    dfssf[[17]][[1]] <- dfssf[[17]][[1]]+1
    dfssf[[17]][[dfssf[[k]][[1]]]] <- (sws[[3]][[i]]+sws[[4]][[i]])/(sws[[3]][[i-1]]+sws[[4]][[i-1]])
  }
}

for(i in 1:17){
  outssf$Mean[[i]] <- (sum(dfssf[i],na.rm = TRUE)-dfssf[[i]][[1]])/(dfssf[[i]][[1]]-1)
  outssf$Var[[i]] <- var(dfssf[[i]][-1],na.rm = TRUE)
}
#计算平均值、方差

dfssf_subset <- dfssf[2:nrow(dfssf),]
p<-0
for(i in 10:17){
  if(shapiro.test(dfssf_subset[[i]])$p.value[[1]] <= 0.05){
    p[i] <- i
  }
}
p
#结果:T10\12\L1\L2不正态，其他均正态

for(i in 10:17){
  outssf$CI1[[i]] <- t.test(dfssf_subset[i])$conf.int[[1]]
  outssf$CI2[[i]] <- t.test(dfssf_subset[i])$conf.int[[2]]
}

outssf <- outssf[-(1:10),]
write.csv(outssf,file = "outssf.csv",row.names = F)

dfplot <- data.frame(matrix(double(),7000,3))
colnames(dfplot) <- c("value","segment","group")
x <- 1
for (k in 11:16) {
  for (i in 1:800){
    dfplot[[1]][[i+(k-11)*800]] <- df2[[k]][[i]]
    dfplot[[2]][[i+(k-11)*800]] <- colnames(df2[k])
    dfplot[[3]][[i+(k-11)*800]] <- "normal"
    x <- x+1
  }
}

for (k in 11:16) {
  for (i in 1:300){
    if(is.na(dfsf_subset[[k]][[i]]) == F){
    dfplot[[1]][[x]] <- dfsf_subset[[k]][[i]]
    dfplot[[2]][[x]] <- colnames(dfsf_subset[k])
    dfplot[[3]][[x]] <- "fracture"
    x <- x+1
    }
  }
}

for (k in 11:16) {
  for (i in 1:300){
    if(is.na(dfssf_subset[[k]][[i]]) == F){
      dfplot[[1]][[x]] <- dfssf_subset[[k]][[i]]
      dfplot[[2]][[x]] <- colnames(dfssf_subset[k])
      dfplot[[3]][[x]] <- "suspected"
      x <- x+1
    }
  }
}

dfplot <- dfplot[(1:5842),]
set.seed(121)
dfplot_group <- dfplot %>% sample_n(size = nrow(dfplot),replace = F)
dfplot_group$layer <- NA
for (i in 1:4090) {
  dfplot_group$layer[[i]] <- "train"
}
for(i in 4091:4674){
  dfplot_group$layer[[i]] <- "valid_test"
}
for(i in 4675:5842){
  dfplot_group$layer[[i]] <- "test"
}

t.test(dfsf_subset,alternative = "less")
dfnr_subsut <- df2[,-(1:10)]
dfnr_subsut <- dfnr_subsut[,-7]
t.test(dfnr_subsut,alternative = "greater")

dfplot <- dfplot[complete.cases(dfplot),]

ggplot(dfplot_group[(dfplot_group$layer %in% "train"),],aes(value,fill = group))+
  geom_histogram(binwidth=0.02,data = subset(dfplot_group[(dfplot_group$layer == "train"),],group == "fracture"),alpha = 0.7)+
  geom_histogram(binwidth=0.02,data = subset(dfplot_group[(dfplot_group$layer == "train"),],group == "suspected"),alpha = 0.7)+
  geom_histogram(binwidth=0.02,data = subset(dfplot_group[(dfplot_group$layer == "train"),],group == "normal"),alpha = 0.7)
  geom_histogram(breaks =
                   seq(0.2,2.8,0.05))
  geom_vline(xintercept = 0.9727609,color = 2,size = 0.7,linetype = "dashed")+
  geom_vline(xintercept = 1.004536,color = 4,size = 0.7,linetype = "dashed")+
  geom_vline(xintercept = 1.004536,color = 4,size = 0.7,linetype = "dashed")+
  geom_vline(xintercept = 1.493117,color = 3,size = 0.7,linetype = "dashed")

ggplot(dfplot_group[(dfplot_group$layer %in% "train"),],aes(value,fill = group))+
  geom_histogram(breaks =
                   seq(0.2,2.8,0.05))+
  facet_wrap(~segment,ncol = 2)

level <- outsf[,1]
outsf$catagory <- factor(outsf$sort,levels = level)

ggplot(outsf,aes(catagory,Mean,))+
  geom_point(size = 3,col = 4)+
  labs(x = "fracture")

dfsf_subset <- dfsf_subset[,11:16]
dfssf_subset <- dfssf_subset[,11:16]
df2 <- df2[,11:16]
x <- 0
for(i in 1:6){
  for(k in 1:292){
    if (is.na(dfsf_subset[[i]][[k]]) == F){
      if(dfsf_subset[[i]][[k]] <= 1.493117){
        x <- x+1
      }
    }
  }
}
x
for(i in 1:6){
  for(k in 1:292){
    if (is.na(dfssf_subset[[i]][[k]]) == F){
      if(dfssf_subset[[i]][[k]] <= 1.493117){
        x <- x+1
      }
    }
  }
}
x
for(i in 1:6){
  for(k in 1:800){
    if (is.na(df2[[i]][[k]]) == F){
      if(df2[[i]][[k]] >= 0.9727609){
        x <- x+1
      }
    }
  }
}
x

x<-t.test(df2[4],dfsf_subset[4])$p.value

outtest <- data.frame(matrix(double(),3,6))
colnames(outtest) <- c("T11","T12","L1","L2","L3","L4")
for(k in 1:6){
      outtest[[k]][[1]] <- t.test(df2[k],dfssf_subset[k])$p.value
      outtest[[k]][[2]] <- t.test(df2[k],dfsf_subset[k])$p.value
      outtest[[k]][[3]] <- t.test(dfssf_subset[k],dfsf_subset[k])$p.value
}

t.test(dfssf_subset[5],dfsf_subset[5])
write.csv(outtest,file = "ttest.csv")

sequence <- order(dfplot$value)
dfplot <- dfplot[sequence,]

x <- 0
y <- 0
for (i in 1:5842) {
  if(dfplot$group[[i]] == "normal"){
    x <- x+1
    if(x == 240){
      y <- dfplot$value[[i]]
    }
  }
}
y
#计算95置信区间

x <- 0
y <- 0
for (i in 1:5842) {
  if(dfplot$group[[i]] == "fracture"){
    x <- x+1
    if(x == 694){
      y <- dfplot$value[[i]]
    }
  }
}
y

x <- 0
y <- 0
for (i in 1:5842) {
  if(dfplot$group[[i]] == "suspected"){
    x <- x+1
    if(x == 295){
      y <- dfplot$value[[i]]
    }
  }
}
y

segment <- c("T11","T12","L1","L2","L3","L4")
y<-0
for(k in 1:6){
  x <- 0
  for (i in 1:nrow(dfplot[dfplot$segment %in% segment[[k]],])) {
    if(dfplot[dfplot$segment %in% segment[[k]],]$group[[i]] == "normal"){
      x <- x+1
      if(x == 40){
        y[k] <- dfplot[dfplot$segment %in% segment[[k]],]$value[[i]]
      }
    }
  }
}
y

group <- c(52,151,276,117,53,45)
for(k in 1:6){
  x <- 0
  for (i in 1:nrow(dfplot[dfplot$segment %in% segment[[k]],])) {
    if(dfplot[dfplot$segment %in% segment[[k]],]$group[[i]] == "fracture"){
      x <- x+1
      if(x == group[[k]]){
        y[k] <- dfplot[dfplot$segment %in% segment[[k]],]$value[[i]]
      }
    }
  }
}
y

group <- c(26,67,111,51,21,20)
for(k in 1:6){
  x <- 0
  for (i in 1:nrow(dfplot[dfplot$segment %in% segment[[k]],])) {
    if(dfplot[dfplot$segment %in% segment[[k]],]$group[[i]] == "suspected"){
      x <- x+1
      if(x == group[[k]]){
        y[k] <- dfplot[dfplot$segment %in% segment[[k]],]$value[[i]]
      }
    }
  }
}
y

nrow(dfplot[(dfplot$segment %in% "L4") & (dfplot$group %in% "suspected"),])

dfplot[dfplot$segment %in% "L4",] %>%
  ggplot(aes(value,fill = group))+
  geom_histogram(breaks =
                   seq(0.2,2.8,0.05))+
  geom_vline(xintercept = 0.9052896,color = 2,size = 0.7,linetype = "dashed")+
  geom_vline(xintercept = 1.0377436,color = 4,size = 0.7,linetype = "dashed")+
  geom_vline(xintercept = 1.550854,color = 3,size = 0.7,linetype = "dashed")+
  labs(title = "L4")


dfplot_group$group2 <- NA
for (i in 1:5842) {
  if(dfplot_group$group[[i]] == "normal"){
    dfplot_group$group2[[i]] <- "normal"
  }
  if(dfplot_group$group[[i]] != "normal"){
    dfplot_group$group2[[i]] <- "fracture"
  }
}
roc1 <- roc(dfplot_group[dfplot_group$group != "suspected" & dfplot_group$layer == "train",]$group,dfplot_group[dfplot_group$group != "suspected" & dfplot_group$layer == "train",]$value)
plot(roc1,col="blue")
roc2 <- roc(dfplot_group[dfplot_group$group != "fracture" & dfplot_group$layer == "train",]$group,dfplot_group[dfplot_group$group != "fracture" & dfplot_group$layer == "train",]$value)
plot(roc2,add=TRUE,col="red")
roc3 <- roc(dfplot_group[dfplot_group$layer == "train",]$group2,dfplot_group[dfplot_group$layer == "train",]$value)
plot(roc3,add=T,col="green")
auc(roc2)

cutoff <- coords(roc2,"best",best.method = c("youden"),
                 ret=c("threshold","sensitivity","specificity"))
cutoff

dfplot_group$label <- NA
for (i in 1:4800) {
  if(dfplot_group[dfplot_group$group == "normal",]$value[[i]] <= 0.9379378){
    dfplot_group[dfplot_group$group == "normal",]$label[[i]] <- "FN"
  }
}
for (i in 1:4800) {
  if(dfplot_group[dfplot_group$group == "normal",]$value[[i]] > 0.9379378){
    dfplot_group[dfplot_group$group == "normal",]$label[[i]] <- "TN"
  }
}
for (i in 1:5842) {
  if(dfplot_group[dfplot_group$group == "fracture",]$value[[i]] > 0.9379378){
    dfplot_group[dfplot_group$group == "fracture",]$label[[i]] <- "FP"
  }
}
for (i in 1:5842) {
  if(dfplot_group[dfplot_group$group == "fracture",]$value[[i]] <= 0.9379378){
    dfplot_group[dfplot_group$group == "fracture",]$label[[i]] <- "TP"
  }
}

Precision = count(dfplot_group[dfplot_group$label == "TP" & dfplot_group$layer == "valid_test",])/(count(dfplot_group[dfplot_group$label == "TP" & dfplot_group$layer == "valid_test",])+count(dfplot_group[dfplot_group$label == "FN" & dfplot_group$layer == "valid_test",]))
Precision

Recall <- count(dfplot_group[dfplot_group$label == "TN" & dfplot_group$layer == "valid_test",])/(count(dfplot_group[dfplot_group$label == "TN" & dfplot_group$layer == "valid_test",])+count(dfplot_group[dfplot_group$label == "FP" & dfplot_group$layer == "valid_test",]))
Recall

F1 <- 2*Precision*Recall/(Precision+Recall)
F1

count(dfplot_group[dfplot_group$label == "FP" & dfplot_group$layer == "valid_test",])
count(dfplot_group[dfplot_group$label == "FN" & dfplot_group$layer == "valid_test",])
count(dfplot_group[dfplot_group$layer == "valid_test" & dfplot_group$group != "normal",])
count(dfplot_group[dfplot_group$layer == "valid_test" & dfplot_group$group == "normal",])

count(dfplot_group[dfplot_group$label == "FP" & dfplot_group$layer == "test",])
count(dfplot_group[dfplot_group$label == "FN" & dfplot_group$layer == "test",])
count(dfplot_group[dfplot_group$layer == "test" & dfplot_group$group != "normal",])
count(dfplot_group[dfplot_group$layer == "test" & dfplot_group$group == "normal",])
Precision = count(dfplot_group[dfplot_group$label == "TP" & dfplot_group$layer == "test",])/(count(dfplot_group[dfplot_group$label == "TP" & dfplot_group$layer == "test",])+count(dfplot_group[dfplot_group$label == "FP" & dfplot_group$layer == "test",]))
Precision

Recall <- count(dfplot_group[dfplot_group$label == "TP" & dfplot_group$layer == "test",])/(count(dfplot_group[dfplot_group$label == "TP" & dfplot_group$layer == "test",])+count(dfplot_group[dfplot_group$label == "FN" & dfplot_group$layer == "test",]))
Recall

range(dfplot[dfplot$group == "fracture",]$value)
t.test(dfsf_subset[1],dfssf_subset[1])
wilcox.test(dfssf_subset$T11,dfsf_subset$T11)
mean(dfplot[dfplot$group == "normal",]$value)
median(dfplot[dfplot$group == "normal",]$value)