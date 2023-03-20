# Projeto com Feedback 1
# Detecção de Fraudes no Tráfego de Cliques em Propagandas de Aplicações Mobile
#
# Nome do Aluno: Victor Hugo Nishitani
#
# Você está desafiado a criar um algoritmo que possa prever se um usuário fará o download de um 
# aplicativo depois de clicar em um anúncio de aplicativo para dispositivos móveis.
#
# Neste projeto, você deverá construir um modelo de aprendizado de máquina para determinar se 
# um clique é fraudulento ou não.
#
# Dicionário de Dados:
# ip: endereço IP do usuário.
# app: ID do aplicativo de marketing.
# device: ID do tipo de dispositivo do usuário (exemplo, iphone 6 plus, iphone 7, huawei mate 7, etc.)
# os: ID da versão do sistema operacional do dispositivo do usuário.
# channel: ID do canal do editor de anúncios.
# click_time: data/hora do clique (UTC).
# attributed_time: se o usuário baixar o aplicativo após clicar em um anúncio, data/hora do download do aplicativo.
# is_attributed: o alvo que deve ser previsto, indicando que o aplicativo foi baixado.


# Configurando o diretório de trabalho
setwd("/Users/nishi/Desktop/FCD/BigDataRAzure/Cap18/Projeto 1")
getwd()

# Carrega os pacotes na sessão R
library(ggplot2)
library(randomForest)
library(e1071)
library(caTools)
library(caret)
library(ROCR)
library(ROSE)


## Etapa 1 - Coletando os Dados
##### Carga dos Dados ##### 

# Carrega o dataset antes da transformação
df <- read.csv("train_sample.csv",sep = ",", header = TRUE, stringsAsFactors = FALSE)



## Etapa 2 - Pré-Processamento
##### Análise Exploratória dos Dados - Limpeza e Organização de Dados ##### 

# Visualizando os dados

# A coluna is_attributed indica se o aplicativo foi baixado ou não, sendo portanto nossa variável target
# 0 indica que o aplicativo não foi baixado (clique fraudulento)
# 1 indica que o aplicativo foi baixado (clique não fraudulento)
View(df)
str(df)
dim(df)

# Verificando se temos valores ausentes
sum(is.na(df))

# A coluna is_attributed está completamente desbalanceada. 
# Temos cerca de 99% de registros para o valor 0 e menos de 1% para o valor 1
prop.table(table(df$is_attributed))

# Essa diferença entre os valores 0 e 1 pode ser vista no gráfico. Então, recomenda-se
# balancear essa coluna, para que não ocorra "Overfitting" e nosso modelo se torne
# ineficaz. Também podemos executar dessa forma e comparar com o modelo balanceado. 
ggplot(df,aes(is_attributed)) + geom_bar(aes(fill = factor(is_attributed)), alpha = 0.5)

# Vamos converter a coluna alvo is_attributed para fator
df$is_attributed <- as.factor(df$is_attributed)

# Vamos converter as colunas click_time e attributed_time para data/hora do calendário
# Obs: Como o problema que queremos resolver não tem relação com o tempo, é provável
# que não utilizaremos essas colunas
df$click_time <-as.POSIXct(strptime(paste(df$click_time, " ", as.character(df$click_time),
                                          ":00:00", sep = ""), "%Y-%m-%d %H:%M:%S"))

df$attributed_time <-as.POSIXct(strptime(paste(df$attributed_time, " ", as.character(df$click_time),
                                          ":00:00", sep = ""), "%Y-%m-%d %H:%M:%S"))

str(df)

# Feature Selection (Seleção de Variáveis)

# Vamos utilizar o modelo randomForest para criar um plot de importância das variáveis
modelo <- randomForest(is_attributed ~ .,
                       data = df, 
                       ntree = 100, nodesize = 10, importance = T)

varImpPlot(modelo)

# Inicialmente vamos selecionar as 5 colunas mais importantes para executar nosso
# modelo (app, ip, channel, click_time e attributed_time), mas posteriormente 
# podemos incluir ou excluir colunas conforme a nossa necessidade



## Etapa 3: Treinando o modelo e Criando o Modelo Preditivo no R

# Vamos dividir os dados em treino e teste, sendo 70% para dados de treino e 
# 30% para dados de teste
set.seed(2)
split = sample.split(df$is_attributed, SplitRatio = 0.70)
dados_treino = subset(df, split == TRUE)
dados_teste = subset(df, split == FALSE)

# Vejamos como está agora a distribuição da nossa coluna alvo nos dados 
# de treino e de teste, ainda continua desbalanceada
prop.table(table(dados_treino$is_attributed))
prop.table(table(dados_teste$is_attributed))

# Verificando o número de linhas
nrow(dados_treino)
nrow(dados_teste)

# Primeiro, vamos criar um modelo de Machine Learning com os dados desbalanceados,  
# e comparar com o resultado após o balanceamento da nossa coluna alvo is_attributed

# Vamos utilizar o método "Random Forest" com as variáveis que selecionamos como
# mais importantes com os dados de treino
modelo_v1 <- randomForest(is_attributed ~ app
                          + ip
                          + channel
                          + attributed_time
                          + click_time, 
                          data = dados_treino, 
                          ntree = 100, 
                          nodesize = 10)

# Imprimindo o resultado
print(modelo_v1)

# Agora fazemos as previsões com o modelo usando dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)



## Etapa 4: Avaliando o modelo

# Criamos a Confusion Matrix e analisamos a acurácia do modelo
# O parâmetro positive = '0' indica que a classe 0 é a positiva, ou seja, indica que sim, 
# o clique é possivelmente fraudulento, pois o aplicativo não foi baixado
caret::confusionMatrix(dados_teste$is_attributed, previsoes_v1)

# Analisando o resultado da Confusion Matrix, temos que nosso modelo do total de 29932 
# cliques possivelmente fraudulentos, acertou 29925 ou 99% dos cliques (0 ou falso negativo)
# e errou apenas 7 ou menos de 1% dos cliques (1 ou falso positivo) que possivelmente eram
# fraudulentos, e o modelo indicou que não eram. De maneira oposta, do total de 68 cliques que 
# eram possivelmente não fraudulentos, devido ao desbalanceamento, note-se que o modelo errou
# mais quando indicou que 60 ou 88% dos cliques (0 ou verdadeiro negativo) não eram fraudulentos,
# mas na realidade eram. E por fim, acertou 8 ou 12% dos cliques (1 ou verdadeiro positivo),
# que possivelmente não eram fraudulentos, ou seja, o aplicativo foi baixado.

# Agora criamos a Curva ROC para encontrar a métrica AUC
roc.curve(dados_teste$is_attributed, previsoes_v1, plotit = T, col = "red")

# Como resultados, nós temos:

# Acurácia = 0.997
# Score AUC = 0.559


# Agora vamos utilizar o ROSE para balancear a coluna is_attributed nos dados de treino e teste, 
# usando a técnica de Oversampling

# Aplicando ROSE em dados de treino e checando a proporção de classes
rose_treino <- ROSE(is_attributed ~ app
                    + ip
                    + channel
                    + os
                    + device,
                    data = dados_treino, 
                    seed = 1)$data

prop.table(table(rose_treino$is_attributed))

# Aplicando ROSE em dados de teste e checando a proporção de classes
rose_teste <- ROSE(is_attributed ~ app 
                   + ip
                   + channel
                   + os
                   + device, 
                   data = dados_teste, 
                   seed = 1)$data

prop.table(table(rose_teste$is_attributed))

# Utilizamos o método "Random Forest" com os dados de treino balanceados com o ROSE
modelo_v2 <- randomForest(is_attributed ~ app
                          + ip
                          + channel
                          + os
                          + device,
                          data = rose_treino, 
                          ntree = 100, 
                          nodesize = 10)

# Imprimindo o resultado
print(modelo_v2)

# Agora fazemos as previsões com o modelo usando dados de teste
previsoes_v2 <- predict(modelo_v2, rose_teste)

# Criamos a Confusion Matrix e analisamos a acurácia do modelo
# O parâmetro positive = '0' indica que a classe 0 é a positiva, ou seja, indica que sim, 
# o clique é fraudulento
caret::confusionMatrix(rose_teste$is_attributed, previsoes_v2)

# Nosso modelo balanceado do total de 15121 dos cliques possivelmente fraudulentos,
# acertou 13307 ou 88% dos cliques (0 ou falso negativo) e errou 1814 dos cliques ou 12%
# (1 ou falso positivo) que possivelmente eram fraudulentos, e o modelo indicou
# que não eram. De maneira oposta, do total de 14879 dos cliques que possivelmente não
# eram fraudulentos, após o balanceamento, note-se que o modelo errou proporcionalmente 
# menos quando indicou que 1999 ou 13% dos cliques (0 ou verdadeiro negativo) não eram 
# fraudulentos, mas na realidade eram. E por fim, acertou 12880 ou 87% dos cliques (1 ou
# verdadeiro positivo), que possivelmente não eram fraudulentos, ou seja, o aplicativo
# foi baixado. 

# Portanto, percebe-se que o modelo balanceado errou mais nos casos de falsos positivos 12%,
# mas dentro de uma taxa aceitável, pois ainda manteve 88% de acertos. E melhorou bastante
# nos casos de verdadeiros positivos acertando 87%, e também, nos verdadeiros negativos 
# errando apenas 13% dos casos

# Agora criamos a Curva ROC para encontrar a métrica AUC
roc.curve(rose_teste$is_attributed, previsoes_v2, plotit = T, col = "red")

# Como resultados, nós temos:

# Acurácia = 0.872
# Score AUC = 0.873

# Veja que mantivemos uma excelente acurácia nos dois modelos (praticamente 99%
# no modelo 1 e 87% no modelo 2), mas aumentamos muito o Score AUC, de 55% para 87%.
# Isso comprova que o modelo_v2 (Balanceado) é melhor e mais estável que o modelo_v1 
# (Desbalanceado).



## Etapa 5: Otimizando o Modelo preditivo

# Por último, vamos tentar Otimizar utilizando outro modelo, também 
# vamos usar os dados de treino e teste balanceados

# Agora criamos o modelo "SVM" usando dados de treino balanceados
modelo_v3 <- svm(is_attributed ~ ., data = rose_treino)

# E fazemos previsões usando dados de teste balanceados
previsoes_v3 <- predict(modelo_v3, rose_teste)

# Vamos verificar a acurácia
caret::confusionMatrix(rose_teste$is_attributed, previsoes_v3)

# Agora criamos a Curva ROC para encontrar a métrica AUC
roc.curve(rose_teste$is_attributed, previsoes_v3, plotit = T, col = "red")

# Como resultados, nós temos:

# Acurácia = 0.879
# Score AUC = 0.880

# Note que conseguimos aumentar levemente o Score AUC, de 87% para 88%, e também a
# acurácia de 87% para 88%. Logo, o modelo_v3 é um pouco melhor do que o modelo_v2.


