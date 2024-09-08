# Regressão Linear

> Notas sobre notação. Por enquanto utilizo a mesma notação que Andrew Ng no curso CS229 em Stanford a não ser que eu diga explicitamente o contrário. Se no futuro eu mudar tentarei ser consistente ao longo de todos os arquivos.

ADEQUAR NOTAÇÃO DEPOIS

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

Com essa equação da reta definida chegamos então ao ponto importante, como determinar essa equação se tudo que temos é um conjunto de pontos? Existem algumas maneiras de fazer isso, a primeira segue um método estatístico conhecido como Mínimos Quadrados Ordinários (OLS, em inglês, *Ordinary Least Squares*) e uma segunda é uma abordagem utilizando o Método de Gradiente.

> AJUSTAR DEPOIS PARA FAZER SENTIDO COM O RESTO DO CAPÍTULO

Nas próximas sessões ambos os métodos para resolver esse problema vão ser explicados tanto de maneira intuitiva quanto a sua definição formal na matemática. O foco das sessões será na explicação do método, porém em quadros de destaque será continuado o exercício proposto acima de prever preço de imóveis usando regressão linear.


> [!NOTE] Exemplo de destaque
> Exemplo de quadro de destaque.

## Definição Formal

> AJUSTAR PARA DEPOIS PARA SEGUIR COISAS QUE VI NO WOOLDRIDGE

Regressão Linear é um método de aprendizado de máquina muito utilizado para resolver uma grande gama de problemas. O objetivo final dela é o mesmo caso estivéssemos usando o método do MQO, traçar uma linha sobre o nosso conjunto de dados de forma a tentar entender como que outros pontos que não estão no nosso conjunto de dados.

Uma regressão linear consegue isso usando uma parte do nosso conjunto de dados como dados de treino e a outra parte como dados de teste. Os dados de treino são usados para treinar o modelo, o algoritmo por trás da regressão usa esses valores para tentar achar a reta ideal que cruza sobre os pontos e os de teste são usados para poder validar os resultados dessa reta em um conjunto de dados ainda não apresentado para o modelo durante o processo de treino.

Um ponto importante no que tange a divisão de variáveis entre treino e teste, é a a forma como o algoritmo "vê" o conjunto de treino. Chamamos a Regressão Linear de um algoritmo do tipo de **aprendizado supervisionado** (em inglês, *supervised learning*) ou seja, ao alimentarmos o modelo com os dados de treino informamos para ele os resultados desse dados de treino para ele poder usar esse par de informações para poder criar a reta que melhor se encaixa aos dados. Uma melhor explicação sobre paradigmas de aprendizado de máquina podem ser encontrados na seção de referência ~~ainda em construção~~.

O processo de como o algoritmo constrói uma reta que melhor se encaixa nos dados é o que separa entre ela poder ser classificada como um método de aprendizado de máquina ou não. MQO é, de forma geral, considerado um método estatístico não vinculado ao aprendizado de máquina, enquanto Método de Gradiente é (para mais detalhes nessa discussão entre estatistica e ML ler a seção referente ~~ainda em construção~~). 

> [!NOTE] ML vs Estatística
Seja pelo método de MQO ou Método do Gradiente (em inglês, *gradient descent*) o objetivo final da regressão linear é o mesmo independente do método escolhido e por isso a escolha do método depende muito mais 

Regressão linear é, então, um algoritmo de aprendizado de máquina de aprendizado supervisionado que tenta melhor encaixar os dados a uma reta. Essa reta terá então o seguinte formato:

$$h_{\theta}(x) = \theta_0 + \theta_1x_1$$

Onde $h$ é a nossa curva de regressão; $\theta_0$ é o intercepto; $\theta_1$ é o coeficiente angular da reta; e $x_1$ o nosso conjunto de dados. $\theta_0$ e $\theta_1$ são muitas vezes chamados de **parâmetros** do nosso modelo. O olhar atento vai notar que essa equação é muito similar a equação da reta que vimos anteriormente nesse capítulo, só temos que mudar os $\theta$ por $m$ e $b$ e é a mesma equação! A manipulação dos valores nos parâmetros nos dá então qualquer reta que pode ser representada num espaço cartesiano. **O nosso objetivo é, então, encontrar o conjunto de $\theta$ que gera a reta que melhor se encaixa no conjunto de dados**.

O intercepto é as vezes chamado de "viés", em termos geométricos, ele nos diz onde que a reta corta o eixo y quando o valor de $x_1$ é igual a 0. Em termos de aprendizado de máquina ele nos diz qual a condição inicial do nosso modelo.

Importante ressaltar que aqui estamos lidando com o caso de que o nosso conjunto de dados tem apenas uma variável, $x_1$, caso ele tenha mais variáveis elas devem ser incluídas na equação geral do modelo da seguinte forma:

$$h_{\theta}(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_nx_n$$

Ou,

$$h_{\theta}(x) = \sum_{i=0}^{n}\theta_ix_i = \theta^Tx$$

O olhar atento vai ver que ignorei que incorporei o intercepto no soma, isso porque estou assumindo $\theta$ e $x$ e como sendo um conjunto de vetores com cada um dos valores $i$, assim convencionamos que $x_0 = 1$. Sendo assim, CONTINUAR DEPOIS

Na equação acima, $n$ é o número de variáveis do modelo. Ao longo dessa primeira parte do capítulo vou lidar somente com o caso de uma variável, as interpretações do caso de múltiplas variáveis será apresentado em outro momento.

## Mínimo Quadrados Ordinários (MQO)

A implementação mais comum para a regressão linear é através do método dos Mínimos Quadrados Ordinários, projetos como o SciKit-Learn em Python ou o GLM.jl em Julia utilizam o MQO como forma de resolver um problema de regressão linear.

Voltemos a nossa equação geral para uma curva de regressão linear, para esse exemplo vamos usar uma regressão linear bivariada:

$$y = \beta_0 + \beta_1x + u$$

## Método de Gradiente

Começamos montando uma regressão linear por método de gradiente assumindo algum valor qualquer para os nossos parâmetros $\theta_0$ e $\theta_1$, e usando os dados que são passados ao modelo mudar esses valores até que eles melhorem se encaixem nas observações, mas como podemos avaliar esse "encaixe" nos dados? Para isso vamos definir algo que chamaremos de Função de Custo (as vezes chamada de "Função de Perda") que medirá o quão perto $h_{\theta}(x^{(i)})$ está do valor verdadeiro $y^{(i)}$. Definiremos a Função de Custo $J(\theta)$ como:

$$J(\theta) = \frac{1}{2}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$$

Embora essa equação pareça assustadora a primeira vista, ela é fácil de compreender caso a quebremos em seus componentes. Uma soma de 1 até $m$, onde $m$ é o número de observações dos dados, do quadrado da diferença entre o valor estimado ($h_{\theta}(x^{(i)})$) e o valor real ($y^{(i)}$). A diferença então nos faz o trabalho de calcular o quão diferente é o nosso valor estimado do real.  

> Aqui vale a ressalva antes de prosseguirmos sobre notação, o sobrescrito $(i$) denota a $i$-ésima observação no conjunto de dados, por exemplo $x^{(2)}$ é a segunda observação.

A questão que pode surgir é onde que os outros elementos dessa conta entram. Por que dividimos o resultado da soma por dois? Por que elevamos ao quadrado? A resposta para a primeira pergunta é, por enquanto, pouco satisfatória, mais a frente, quando tivermos que manipular essa equação, a matemática se torna mais fácil quando dividimos o resultado por dois, tal operação não afeta o resultado final, só torna os cálculos mais fáceis. Já a segunda pergunta é mais interessante. Mais a frente no capítulo VAI SABER QUAL veremos que regressão linear é um caso específico de toda uma família de modelos chamada Modelos Lineares Generalizados (do inglês General Linear Models, GLM) e nesse momento a existência do ao quadrado ficará mais clara.

Tendo em mãos uma função que podemos usar para avaliar o resultado das nossas previsões enquanto treinamos o modelo, queremos então escolher um conjunto de $\theta$ de forma a minimizar $J(\theta)$. **A questão do processo usado para fazer isso é a central dessa seção**. Vamos supor que temos valores iniciais para o conjunto de $\theta$s, e para fins de simplificação vamos supor que temos apenas uma variável e o intercepto, logo $\theta_1$ e $\theta_0$, o nosso algoritmo do método de gradiente vai iterar sobre os nossos valores de $\theta$, atualizando seu valor a cada nova iteração. Para cada $\theta_j$ no nosso problema ($\theta_0$, $\theta_1$...) com um valor inicial qualquer, a cada iteração ele é atualizado dada a seguinte fórmula:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)$$

Expandindo a derivada parcial temos que,

$$\theta_j := \theta_j - \alpha \sum_{i=1}^m(y^{(i)}-h_{\theta}(x^{(i)})x_j^{(i)} \space \text{, para cada j}$$

Onde $\alpha$, a chamado "taxa de aprendizagem", é um hiperparâmetro que define o quão rápido ou devagar o nosso algoritmo vai convergir para uma resposta, mais a frente será um pouco mais comentado como ele é escolhido. No lado esquerdo da equação temos a derivada parcial com respeito a $\theta_j$ da função de custo, caso tenham interesse, a expansão dela está logo abaixo.

> [!NOTE] Notação
> Eu uso o símbolo $:=$ em vez de somente um símbolo igual porque estamos atribuindo um valor a $\theta_j$ e não afirmando que ele é igual a algo. É o mesmo conceito de atribuição em ciência da computação como por exemplo $a := a + 1$ .

$$
\begin{align}
\frac{\partial}{\partial \theta_j} J(\theta) &= \frac{\partial}{\partial \theta_j}\frac{1}{2}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2 \newline
&= 2 \cdot \frac{1}{2} (x)-y)^2 \cdot \frac{\partial}{\partial \theta_j} (h_{\theta}(x)-y)^2 \newline
&= (h_{\theta}(x)-y)^2 \cdot \frac{\partial}{\partial \theta_j} \left(\sum_{i=0}^n\theta_ix_i-y \right) \newline
&= (h_{\theta}(x)-y)x_j
\end{align}
$$



> [!example]- Sobre a derivada parcial
> $$
> \begin{align}
> \frac{\partial}{\partial \theta_j} J(\theta) &= \frac{\partial}{\partial \theta_j}\frac{1}{2}((h_{\theta}(x)-y)^2 \newline
> &= 2 \cdot \frac{1}{2} (x)-y)^2 \cdot \frac{\partial}{\partial \theta_j} (h_{\theta}(x)-y)^2 \newline
> &= (h_{\theta}(x)-y)^2 \cdot \frac{\partial}{\partial \theta_j} \left(\sum_{i=0}^n\theta_ix_i-y \right) \newline
> &= (h_{\theta}(x)-y)x_j
> \end{align}
> $$
> Há três pontos que podem pedir uma explicação a mais. O primeiro está na segunda linha onde foi usada a regra da cadeia e por isso a a diferença do valor estimado e do real aparece duas vezes. O segundo ponto é na terceira linha, onde substituímos a função $h_{\theta}(x)$ pela sua definição, a somatória dos termos e parâmetros. O último é derivada parcial da soma, para simplificar o exemplo contamos que há somente uma *feature*, $x$, por isso o resultado acaba sendo somente $x_j$.








---

## Referências

https://www.youtube.com/watch?v=n03pSsA7NtQ

https://github.com/julia4ta/tutorials/tree/master

https://github.com/maxim5/cs229-2018-autumn

https://www.youtube.com/watch?v=4b4MUYve_U8