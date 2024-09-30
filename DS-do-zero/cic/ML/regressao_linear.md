# Regressão Linear

> Notas sobre notação. Por enquanto utilizo a mesma notação que Andrew Ng no curso CS229 em Stanford a não ser que eu diga explicitamente o contrário. Se no futuro eu mudar tentarei ser consistente ao longo de todos os arquivos.

ADEQUAR NOTAÇÃO DEPOIS

Digamos que tenhamos um conjunto de dados com duas informações: o preço de venda de casas em uma certa região, e a metragem do local (para fins desse exemplo digamos que os valores estão em metros quadrados). Podemos expor esse conjunto graficamente através do seguinte gráfico.

![preco_x_metro](../../_images/houseprices_scatter.png "preco_x_metro")
*Gráfico 1: Preço de Imóvel Vs. Metragem. Dados por julia4ta no GitHub.*

Essas informações nos dizem o preço de todas as casas já vendidas na região, e, para fins de simplificação, vamos supor que somente a metragem influência no preço de uma casa. Vamos então ignorar qualquer outro fator que pudesse afetar no valor de venda da propriedade coisas como: quantidade de banheiros, quartos, vizinhança, etc. Por enquanto, vamos focar somente na metragem e preço.

Agora, vamos imaginar outro cenário. João quer vender a sua casa de 350m² e gostaria de saber a sua opinião sobre qual valor, ele sabe que se colocar uma valor muito alto a casa não será vendida e se colocar um valor baixo ele estará perdendo dinheiro. Ou seja, com base no que você sabe sobre o preço de venda de casas, que valor diria par ao João que é o ideal? Ou ainda, se fosse outra pessoa, Maria, que lhe fizesse essa mesma pergunta só que a casa dela 100m², que valor diria para ela? De forma genérica podemos então formular a seguinte pergunta: 

> Dada uma casa de metragem $A$ qual será seu valor de venda $p$?

Em um mundo ideal, temos uma conhecimento perfeito de como cada cada metragem influência na venda de uma casa, mas, se vivêssemos nesse mundo perfeito, computadores seriam tão poderosos em fazer previsões que o mundo que conhecemos hoje provavelmente não existiria! O que podemos fazer, então, é dar um "chute", uma previsão, analisar as informações que temos e tentar extrapolar dela uma resposta para a nossa pergunta principal.

Uma análise rudimentar dos dados parece nos mostrar que conforme a metragem de uma casa aumenta (eixo X), seu preço também aumenta (eixo Y). Podemos ilustrar isso com uma linha imaginário sobre os dados, como na Figura 1. Claro, isso é somente uma suposição com base no que temos de informação, não sabemos se é verdade ou não, e só podemos afirmar isso porque ao modelar esses dados ignoramos todos os outros fatores que influenciam no preço de uma casa como dito anteriormente.

![preco_x_metro_tendencia](../../_images/houseprices_scatter_line.png "preco_x_metro_tendencia")
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

> [!example] ML vs Estatística
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

Onde $y$ é a nossa curva de regressão, os $\beta$s são os nossos parâmetros e $u$ o erro. Chamos esse modelo de bivariada pois relaciona duas variáveis, $y$ e $x$.

Fazemos algumas suposições sobre a natureza estatística desse modelo para que ele possa ser verdade, a primeira é que a esperança do erro é 0, ou seja, o seu valor médio é zero. Isso porque consideramos que, dado que o intercepto $\beta_0$ esteja incluído na equação, a perturbação causada pelo erro já pe corrigida por este.

$$\text{E}(u)=0$$

Além disso há outro ponto importante a se considerar: como $u$ e $x$ estão relacionadas. Podemos pensar em termos de correlação, contudo uma propriedade interessante desta é que as vezes não é correlação com $x$, mas sim com uma função de $x$ (digamos $x^2$), o que pode causar problemas na hora de realizarmos uma regressão. Para isso temos que pensar então em **qual é o valor esperado de $u$, dado $x$**.

$$\text{E}(u|x)=E(u)=0$$

Em outras palavras queremos pensar que, dado uma distribuição qualquer de $x$, o valor médio de $u$ é o mesmo, zero. Por exemplo, que não importa se uma pessoa estudou 2, 4, ou 10 anos, supomos que a aptidão natural média para um emprego qualquer é zero na hora de tentarmos criar um modelo sobre como educação influencia renda.

Por fim, assumimos também a seguinte condição:

$$\text{Cov}(u, x)=E(ux)=0$$

Ou seja, assumimos que não há correlação entre as variáveis explicativas e o erro, ou seja, o "andar" da variável $x$, nada nos diz sobre o "andar" da variável $u$.

Aqui nos apresenta o problema, $x$ é a nossa variável independente, $y$, o que queremos estimar, $u$ o erro. Já $\beta_0$ e $\beta_1$, como podemos obter bons valores estimados para eles, respectivamente $\hat{\beta}_0$ e $\hat{\beta}_1$?

Para isso que utilizamos o método dos Mínimos Quadrados Ordinários, ele nos permite criar uma boa estimativa para esses valores que se encaixa nos nosso conjunto de dados. Para isso, a primeira etapa é modifiicar a nossa equação principal e pensar em termos do valor esperado.

$$
\begin{align}
y &= \beta_0 + \beta_1x + u \newline
y - \beta_0 - \beta_1x &= u \newline
\text{E}(y - \beta_0 - \beta_1x) &= \text{E}(u) \newline
\text{E}(y - \beta_0 - \beta_1x) &= 0
\end{align}
$$

Ou ainda, dado que $\text{E}(xu)=0$,

$$\text{E}(x(y-\beta_0-\beta_1x)) = 0$$

Dada então uma amostra de dados podemos escolher o melhor $\hat{\beta}_0$ e $\hat{\beta}_1$ para se encaixar nesses dados e resolver a equação acima. A esperança é somente a média das observações então podemos reescrever a equação acima como:

$$
n^{-1}\sum_{i=1}^n x_i(y_i-\hat{\beta}_0-\hat{\beta}_1x_i)=0
$$

Podemos tentar resolver essa equação para $\hat{\beta}_0$ ou $\hat{\beta}_1$, só que temos somente uma equação, temos que achar uma outra com os mesmo termos para fazer a substituição. Para isso vamos aplicar a mesma regra de expandir a esperança a outra equação.

$$
\begin{align}
\text{E}(y - \beta_0 - \beta_1x) &= 0 \newline
n^{-1}\sum_{i=1}^n y_i-\hat{\beta}_0-\hat{\beta}_1x_i &= 0 \newline
n^{-1}\sum_{i=1}^n y_i - n^{-1}\sum_{i=1}^n \beta_0 - n^{-1}\sum_{i=1}^n \beta_1x_i &= 0 \newline
\overline{y} - n^{-1} n \beta_0 - \beta_1 \overline{x} &= 0 \newline
\overline{y} &= \beta_0 + \beta_1 \overline{x}
\end{align}
$$

Em que $\overline{y}$ é a média amostral de $y$, e igualmente para $x$. O que nos permite escrever $\beta_0$ em termos dos outros componentes.

$$\hat{\beta}_0 = \overline{y}-\hat{\beta}_1 \overline{x}$$

Assim podemos susbtituir na equação de antes.

$$
n^{-1}\sum_{i=1}^n x_i(y_i-(\overline{y}-\beta_1 \overline{x})-\hat{\beta}_1x_i)=0
$$

Um ponto interessante aqui é que $n$, o tamanho da amostra, é sempre maior que zero. Logo para a identidade acima se manter só precisamos que a parte da somatória seja igual a zero, por isso podemos suprimir o $n^{-1}$, não fará diferença nos cálculos e o leitor é convidado a confirmar isso.

Suprimindo o tamanho da amostra e rearranjando a equação temos que:

$$
\sum_{i=1}^n x_i(y_i-\overline{y}) = \hat{\beta}_1 \sum_{i=1}^n x_i(x_i-\overline{x})
$$

Graças a algumas propriedades da soma podemos rearranjar a equação como:

$$
\sum_{i=1}^n(x_i-\overline{x})(y_i-\overline{y})=\beta_1\sum_{i=1}^n(x_i-\overline{x})^2
$$

Ou seja, desde que $\sum_{i=1}^n(x_i-\overline{x})^2>0$, podemos estimar $\beta_1$ como sendo:

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^n(x_i-\overline{x})(y_i-\overline{y})}{\sum_{i=1}^n(x_i-\overline{x})^2}
$$

E tendo uma estimativa para $\beta_1$, podemos usar a fórmula que usamos acima para susbtituir o valor de $\beta_0$ para determiná-lo, $\hat{\beta}_0 = \overline{y}-\hat{\beta}_1 \overline{x}$.

E com isso temos a nossa estimativa tanto para o intercepto quanto para o coeficiente linear da reta!

Um olhar atento sobre a nossa fórmula para a estimativa de $\hat{\beta}_1$ mostra que o numerador da equação é a covariância amostral entre os nossos parâmetros e *target*, enquanto o denominador é a variância de $x$.

### Regressão por MQO com múltiplas variáveis

Até agora pensamos no MQO com somente uma variável, contudo, muitas vezes esse não é o caso. Temos diversas variáveis no modelo.

## Método de Gradiente

Começamos montando uma regressão linear por método de gradiente assumindo algum valor qualquer para os nossos parâmetros $\theta_0$ e $\theta_1$, e usando os dados que são passados ao modelo mudar esses valores até que eles melhorem se encaixem nas observações, mas como podemos avaliar esse "encaixe" nos dados? Para isso vamos definir algo que chamaremos de Função de Custo (as vezes chamada de "Função de Perda") que medirá o quão perto $h_{\theta}(x^{(i)})$ está do valor verdadeiro $y^{(i)}$. Definiremos a Função de Custo $J(\theta)$ como:

$$J(\theta) = \frac{1}{2}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$$

Embora essa equação pareça assustadora a primeira vista, ela é fácil de compreender caso a quebremos em seus componentes. Uma soma de 1 até $m$, onde $m$ é o número de observações dos dados, do quadrado da diferença entre o valor estimado ($h_{\theta}(x^{(i)})$) e o valor real ($y^{(i)}$). A diferença então nos faz o trabalho de calcular o quão diferente é o nosso valor estimado do real.  

> Aqui vale a ressalva antes de prosseguirmos sobre notação, o sobrescrito $(i$) denota a $i$-ésima observação no conjunto de dados, por exemplo $x^{(2)}$ é a segunda observação.

A questão que pode surgir é onde que os outros elementos dessa conta entram. Por que dividimos o resultado da soma por dois? Por que elevamos ao quadrado? A resposta para a primeira pergunta é, por enquanto, pouco satisfatória, mais a frente, quando tivermos que manipular essa equação, a matemática se torna mais fácil quando dividimos o resultado por dois, tal operação não afeta o resultado final, só torna os cálculos mais fáceis. Já a segunda pergunta é mais interessante. Mais a frente no capítulo VAI SABER QUAL veremos que regressão linear é um caso específico de toda uma família de modelos chamada Modelos Lineares Generalizados (do inglês General Linear Models, GLM) e nesse momento a existência do ao quadrado ficará mais clara.

Tendo em mãos uma função que podemos usar para avaliar o resultado das nossas previsões enquanto treinamos o modelo, queremos então escolher um conjunto de $\theta$ de forma a minimizar $J(\theta)$. **A questão do processo usado para fazer isso é a central dessa seção**. Vamos supor que temos valores iniciais para o conjunto de $\theta$s, e para fins de simplificação vamos supor que temos apenas uma variável e o intercepto, logo $\theta_1$ e $\theta_0$, o nosso algoritmo do método de gradiente vai iterar sobre os nossos valores de $\theta$, atualizando seu valor a cada nova iteração. Para cada $\theta_j$ no nosso problema ($\theta_0$, $\theta_1$...) com um valor inicial qualquer, a cada iteração ele é atualizado dada a seguinte fórmula:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)$$

Expandindo a derivada parcial temos que,

$$\theta_j := \theta_j - \alpha \left( \sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)} \right) \space \text{, para cada j}$$

> [!NOTE] Notação
> Eu uso o símbolo $:=$ em vez de somente um símbolo igual porque estamos atribuindo um valor a $\theta_j$ e não afirmando que ele é igual a algo. É o mesmo conceito de atribuição em ciência da computação como por exemplo $a := a + 1$ .

Onde $\alpha$, a chamado "taxa de aprendizagem", é um hiperparâmetro que define o quão rápido ou devagar o nosso algoritmo vai convergir para uma resposta, na prática colocamos ele como sendo igual a 0,01 e vemos se encontramos uma curva que melhor se encaixa nos dados, se não continuamos tentando; uma dica é usar intervalos de 0,01 então testar 0,02, 0,03, etc. No lado esquerdo da equação temos a derivada parcial com respeito a $\theta_j$ da função de custo, caso tenham interesse, a expansão dela está logo abaixo.

Atribuição de $\theta_j$ definida acima nos dá então um algoritmo que podemos seguir. Para cada $\theta$, ou parâmetro, do nosso modelo repetimos aquela atribuição de valor até a série convergir para algum valor. Importante ressaltar que o ajuste é feito para todos os parâmetros ao mesmo tempo para depois se checar a convergência. Por exemplo, com dois parâmetros $\theta_0$ e $\theta_1$ se realiza a atualização de valor de ambos e se verifica se o conjunto dos dois convergiram para algum valor de tal forma que a cada nova iteração não há mudanças relevantes no valor do parâmetro.

Esse último ponto é importante "cada nova iteração não há mudanças relevantes no valor do parâmetro", sabemos que conforme a série convergir aos valores ótimos de $\theta_j$ as mudanças a cada iteração serão cada vez menores justamente pela formulação do fator de erro, $(h_{\theta}(x^{(i)})-y^{(i)})$, conforme nosso algoritmo fornece um valor próximo do real, esse valor diminuí assim como o nosso "fator de ajuste" (tudo aquilo que é multiplicado pelo $\alpha$). De forma oposto, erros maiores fazem com que os ajustes a cada iteração sejam maiores.

> [!abstract]- Sobre a derivada parcial
> $$
> \begin{align}
> \frac{\partial}{\partial \theta_j} J(\theta)
> &= \frac{\partial}{\partial \theta_j}\frac{1}{2}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2 \newline
> &= 2 \cdot \frac{1}{2} \sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)}) \cdot \frac{\partial}{\partial \theta_j} \sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)}) \newline
> &= \sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)}) \cdot \frac{\partial}{\partial \theta_j}  \sum_{i=1}^{m} \left(\sum_{k=0}^n \theta_k x_k^{(i)}-y^{(i)} \right) \newline
> &= \sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)}) \cdot \sum_{i=1}^{m} \left(\frac{\partial}{\partial \theta_j} \sum_{k=0}^n \theta_k x_k^{(i)}-y^{(i)} \right) \newline
> &= \sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})\sum_{i=1}^{m}x_j^{(i)} \newline
> &= \sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}
> \end{align}
> $$
> Embora a formulação acima pareça razoavelmente complicada, ela é relativamente simples para aqueles com alguma familiaridade com derivadas parciais, apenas dois pontos merecem atenção. O primeiro é na linha quatro, onde colocamos a soma sobre todas as observações (aquela com limite superior $m$) "fora" da derivada. Isso se dá por causa de uma regra que diz que a derivada de somas é a soma de derivadas. Além disso, na última linha simplificamos o produto das somas para ser a soma dos produtos.
> Outros dois detalhes que merecem atenção é na linha três onde substituo $h_{\theta}$ por sua definição usando uma soma como vimos anteriormente; e na linha cinco onde a derivada parcial de $J(\theta)$ é simplificada a $x_j$, isso é facilmente verificável pois o único termo que depende do $\theta$ em questão é $x_j$.

Esse algoritmo descrito acima é o nosso próprio [método de gradiente](../../matematica/otimizadores/metodo_gradiente.md)! Essa versão em específica é método do gradiente em lote, ou descida do gradiente em lote (do inglês, *batch gradient descent*). Existem outros algoritmos que funcionam na mesmo lógica de descida pelo gradiente como o descida de gradiente estocástico (do inglês, *stochastic gradient descent*) enquanto o aplicado acima precisa percorrer todo o conjunto de dados de teste para dar "um passo", ou seja, ajustar os parâmetros, o método estocástico faz a correção do valor a cada novo valor do conjunto de treino. Por percorrer todo o conjunto de dados de treino, a forma em lote costuma demandar mais poder computacional, mas, na forma apresentada aqui, costuma encontrar o máximo global sem cair em um local e ficar preso ali; por outro lado o estocástico demanda muito menos poder computacional para achar um bom valor de $\theta$, mas nunca costuma encontrar o ponto ótimo e fica oscilando em volta do máximo global.
De forma geral, o método estocástico acaba sendo preferido por convergir mais rapidamente e fornecer resultados bons o suficientes para a maiorias dos casos.

> [!NOTE] Notação
> Na do método de gradiente subtraímos do nosso valor de $\theta$ o $\alpha$ em vez adicionar porque nesse caso estaríamos indo para o lado oposto de um mínimo global, um máximo global. 

POSSO ELABORAR SOBRE NÃO TER MAXIMO LOCAL COM O QUE O ANDREW DIZ EM 34 MINUTOS DA AULA
### Python

Podemos implementar o algoritmo da seguinte forma

### Julia

Podemos implementar o algoritmo da seguinte forma

## Equações normais

As formas que vimos até agora de determinar os parâmetros de uma curva linear de regressão para um dado conjunto de treinamento são eficazes, mas cada uma vem com um custo. Embora o mínimo quadrados ordinários me dê a curva ideal, ele ainda envolve muitas somas sobre percorrermos todo o conjunto de dados somando a diferença dos quadrados, algo que pode escalar computacionalmente dependendo do tamanho do conjunto de treino. O mesmo vale para o método de gradiente, sendo que esse as vezes não nos dá o valor ótimo e sim um que é bom o suficiente. Contudo, podemos pensar em um método que usando o conjunto de treino pode nos dar o valor ótimo dos parâmetros diminuindo a carga computacional, chamaremos esse método de **equações normais**.

Dado um conjunto de treino com $m$ observações e $n$ parâmetros podemos construir uma matriz $X_{m \times n}$ (tecnicamente ela é $m \times n + 1$ se contarmos o intercepto, $\theta_0$, mas para fins de demonstração vou supor que ele é zero pois isso não altera a demonstração).

$$
X =
\left[
\begin{array}{c}
    \textemdash \space (x^{(1)})^T \space \textemdash \\
    \textemdash \space (x^{(2)})^T \space \textemdash \\
    \vdots \\
    \textemdash \space (x^{(m)})^T \space \textemdash \\
\end{array}
\right]
$$

De forma similar podemos definir um vetor coluna com tamanho $n$ (o número de parâmetros) que contém todos os nossos $\theta$.

$$
\theta =
\left[
\begin{array}{c}
    \theta_1 \\
    \theta_2 \\
    \vdots \\
    \theta_n \\
\end{array}
\right]
$$

Logo, podemos multiplicar um pelo outro, só que $(x^{(i)})^T\theta$ é a própria previsão do nosso modelo, $h_{\theta}(x^(i))$, então temos que:

$$
X\theta =
\left[
\begin{array}{c}
    \textemdash \space (x^{(1)})^T \space \textemdash \\
    \textemdash \space (x^{(2)})^T \space \textemdash \\
    \vdots \\
    \textemdash \space (x^{(m)})^T \space \textemdash \\
\end{array}
\right]
\cdot
\left[
\begin{array}{c}
    \theta_1 \\
    \theta_2 \\
    \vdots \\
    \theta_n \\
\end{array}
\right]
=
\left[
\begin{array}{c}
    (x^{(1)})^T\theta \\
    (x^{(2)})^T\theta \\
    \vdots \\
    (x^{(m)})^T\theta \\
\end{array}
\right] \newline \newline
=
\left[
\begin{array}{c}
    h_{\theta}(x^{(1)}) \\
    h_{\theta}(x^{(2)}) \\
    \vdots \\
    h_{\theta}(x^{(n)}) \\
\end{array}
\right]
$$

Seguindo essa mesma linha de representar nossos elementos da equação como vetores, definimos $\vec{y}_m$ como sendo o vetor coluna de tamanho $m$ com os *target*s, isso é o valor real, para o nosso dataset de treino.

$$
\vec{y} = 
\left[
    \begin{array}{c}
        y^{(1)} \\
        y^{(2)} \\
        \vdots \\
        y^{(m)} \\
    \end{array}
\right]
$$

E se subtrairmos $X\theta$ de $\vec{y}$ temos:

$$
\begin{align}
X\theta - \vec{y} &=
\left[
\begin{array}{c}
    (x^{(1)})^T\theta \\
    (x^{(2)})^T\theta \\
    \vdots \\
    (x^{(m)})^T\theta \\
\end{array}
\right]
-
\left[
    \begin{array}{c}
        y^{(1)} \\
        y^{(2)} \\
        \vdots \\
        y^{(m)} \\
    \end{array}
\right] \newline\newline
&=
\left[
\begin{array}{c}
    (x^{(1)})^T\theta - y^{1} \\
    (x^{(2)})^T\theta - y^{2} \\
    \vdots \\
    (x^{(m)})^T\theta - y^{m} \\
\end{array}
\right]
\end{align}
$$

O que é muito similar da nossa equação da função de custo $J(\theta) = \frac{1}{2}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$

Logo, usando a propriedade de que para um vetor $\vec{v}$ qualquer de tamanho $n$, temos que $\vec{v}^T\vec{v} = \sum^n_{i=1} \vec{v}_i^2$ (propriedade esta, também chamada de norma euclidiana). Podemos expressar a função de custo $J(\theta)$ em forma de matrizes.

$$\frac{1}{2}(X\theta - \vec{y})^T(X\theta-\vec{y}) = \frac{1}{2}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$$

Dessa forma podemos minimizar a função de custo $J(\theta)$ derivando-a e igualando-a a zero. Para isso primeiro devemos achar a derivada em relação a $\theta$ da função.

$$
\begin{align}
\nabla_{\theta}J(\theta) &= \nabla_{\theta} \frac{1}{2}(X\theta - \vec{y})^T (X\theta - \vec{y}) \newline
&= \frac{1}{2}\nabla_{\theta}(X^T\theta^T X\theta - X^T\theta^T \vec{y} - \vec{y}^T X\theta + \vec{y}^T\vec{y}) \newline
&= \frac{1}{2}\nabla_{\theta} \text{tr}(X^T\theta^T X\theta - X^T\theta^T \vec{y} - \vec{y}^T X\theta + \vec{y}^T\vec{y}) \newline
&= \frac{1}{2}\nabla_{\theta} (\text{tr}(X^T\theta^T X\theta) - 2 \cdot\text{tr}(\vec{y}^TX\theta) - \text{tr}(\vec{y}^T\vec{y})) \newline
&= \frac{1}{2} (X^TX\theta + X^TX\theta - 2\vec{y}X^T) \newline
&= X^TX\theta-X^T\vec{y}
\end{align}
$$

Algumas explicações são necessárias para essa derivação. Na terceira etapa usamos o fato de que o traço de um escalar é o próprio escalar; na quarta etapa usei o fato de que $\text{tr}(A^T) = \text{tr}(A)$ então pude agregar aqueles dois termos; na sexta etapa me utilizo das propriedades da derivada de matrizes e do traço especificamente a propriedade 5 (sendo $A^T=\theta$, $B=B^T=X^TX$, e $C=I_{n \times n}$, sendo $I$ a matriz identidade) e a propriedade 1; por fim na mesma quinta etapa desconsidero o termo $\vec{y}^T\vec{y}$ pois para a derivada em relação a $\theta$ ele é constante.

Temos então que a derivada de $J(\theta)$ com respeito a $\theta$ é $\nabla_{\theta}J(\theta) = X^TX\theta-X^T\vec{y}$

Para minimizarmos, achar o mínimo local, definimos ela como sendo igual a zero, logo

$$
\begin{align}
0 &= X^TX\theta-X^T\vec{y} \newline
X^TX\theta & = X^T\vec{y}
\end{align}
$$

Reorganizando a equação temos então que:

$$\theta = (X^TX)^{-1}X^T\vec{y}$$

E com isso temos uma equação que nos dá os coeficientes ótimos com base no nosso conjunto de dados de treino!

## Avaliando métricas

---

## Referências

https://www.youtube.com/watch?v=n03pSsA7NtQ

https://github.com/julia4ta/tutorials/tree/master

https://github.com/maxim5/cs229-2018-autumn

https://www.youtube.com/watch?v=4b4MUYve_U8

wooldrige econometria