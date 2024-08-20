# Regressão Linear

Digamos que tenhamos um conjunto de dados com duas informações: o preço de venda de casas em uma certa região, e a metragem do local (para fins desse exemplo digamos que os valores estão em metros quadrados). Podemos expor esse conjunto graficamente através do seguinte gráfico.

![](images/houseprices_scatter.png)
*Gráfico 1: Preço de Imóvel Vs. Metragem. Dados por julia4ta no GitHub.*

Essas informações nos dizem o preço de todas as casas já vendidas na região, e, para fins de simplificação, vamos supor que somente a metragem influência no preço de uma casa. Vamos então ignorar qualquer outro fator que pudesse afetar no valor de venda da propriedade coisas como: quantidade de banheiros, quartos, vizinhança, etc. Por enquanto, vamos focar somente na metragem e preço.

Agora, vamos imaginar outro cenário. João quer vender a sua casa de 350m² e gostaria de saber a sua opinião sobre qual valor, ele sabe que se colocar uma valor muito alto a casa não será vendida e se colocar um valor baixo ele estará perdendo dinheiro. Ou seja, com base no que você sabe sobre o preço de venda de casas, que valor diria par ao João que é o ideal? Ou ainda, se fosse outra pessoa, Maria, que lhe fizesse essa mesma pergunta só que a casa dela 100m², que valor diria para ela? De forma genérica podemos então formular a seguinte pergunta: 

> Dada uma casa de metragem $A$ qual será seu valor de venda $p$?

Em um mundo ideal, temos uma conhecimento perfeito de como cada cada metragem influência na venda de uma casa, mas, se vivêssemos nesse mundo perfeito, computadores seriam tão poderosos em fazer previsões que o mundo que conhecemos hoje provavelmente não existiria! O que podemos fazer, então, é dar um "chute", uma previsão, analisar as informações que temos e tentar extrapolar dela uma resposta para a nossa pergunta principal.

Uma análise rudimentar dos dados parece nos mostrar que conforme a metragem de uma casa aumenta (eixo X), seu preço também aumenta (eixo Y). Podemos ilustrar isso com uma linha imaginário sobre os dados, como na Figura 1. Claro, isso é somente uma suposição com base no que temos de informação, não sabemos se é verdade ou não, e só podemos afirmar isso porque ao modelar esses dados ignoramos todos os outros fatores que influenciam no preço de uma casa como dito anteriormente.

![[houseprices_scatter_line.png]]
*Figura 1: Tendência da relação entre preço e imóvel.*

Vamos chamar essa linha de *curva de regressão*, mais a frente explico porque do nome. Podemos dizer, então, para João e Maria que a estimativa do preço ideal para que eles vendam a casa está em algum lugar na curva de regressão. Matematicamente, podemos usar a fórmula da reta num plano cartesiano para mostrar a nossa curva:

$$y = b + mx$$

Ou ajustando os símbolos para os do nosso problema:

$$p = b + m*A$$

Onde $p$ é o preço da casa, $A$ é a metragem dela e $b$ e $m$ são informações para a construção da curva de regressão. $m$ controla a inclinação da curva, valores mais baixos deixam ela inclinada e vice-versa; no nosso cenário de casas podemos imaginar $m$ como sendo o quão "sensível" é o preço de uma casa a metragem, caso tenhamos um $m$ muito alto, mesmo um aumento pequeno em metragem se reflete em um grande aumento de preço.

Com essa equação da reta definida chegamos então ao ponto importante, como determinar essa equação se tudo que temos é um conjunto de pontos? Existem duas maneiras de fazer isso, a primeira segue um método estatístico conhecido como Mínimos Quadrados Ordinários (OLS, em inglês, *Ordinary Least Squares*) e o segundo é uma abordagem utilizando aprendizado de máquina.

Nas próximas sessões ambos os métodos para resolver esse problema vão ser explicados tanto de maneira intuitiva quanto a sua definição formal na matemática. O foco das sessões será na explicação do método, porém em quadros de destaque será continuado o exercício proposto acima de prever preço de imóveis usando regressão linear.


> [!NOTE] Exemplo de destaque
> Exemplo de quadro de destaque.

## Mínimo Quadrados Ordinários

## Aprendizado de Máquina

Regressão Linear é um método de aprendizado de máquina muito utilizado para resolver uma grande gama de problemas. O objetivo final dela é o mesmo caso estivéssemos usando o método do MQO, traçar uma linha sobre o nosso conjunto de dados de forma a tentar entender como que outros pontos que não estão no nosso conjunto de dados.

Uma regressão linear consegue isso usando uma parte do nosso conjunto de dados como dados de treino e a outra parte como dados de teste. Os dados de treino são usados para treinar o modelo, o algoritmo por trás da regressão usa esses valores para tentar achar a reta ideal que cruza sobre os pontos e os de teste são usados para poder validar os resultados dessa reta em um conjunto de dados ainda não apresentado para o modelo durante o processo de treino.

---
## Referências

https://www.youtube.com/watch?v=n03pSsA7NtQ

https://github.com/julia4ta/tutorials/tree/master

https://github.com/maxim5/cs229-2018-autumn

https://www.youtube.com/watch?v=4b4MUYve_U8