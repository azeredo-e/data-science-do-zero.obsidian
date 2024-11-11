# Regressão Logística

Vamos supor que você trabalha em um laboratório médico e se quer criar uma forma de identificar se nódulos são câncer ou não. Para isso se pensa em medir eles e tentar através de seu tamanho alimentar algum algoritmo que pudesse determinar a probabilidade de ser maligno ou não. Uma primeira tentativa pode ser de tentar usar um método de regressão como a regressão linear ao conjunto de dados, gerar um curva de regressão sobre os dados e avaliar o resultado, se for maior que 0.5, ou qualquer outro valor limite, definir como verdadeiro, caso contrário não.

Vamos tentar ilustrar isso com um exemplo. Digamos que tenhamos o seguinte conjunto de dados sobre o tamanho de nódulos, no eixo x temos o tamanho deles e no eixo Y definimos um valor binário como o nosso target, se o nódulo é maligno ou não, 0 ou 1.

INSERIR AQUI O GRÁFICO QUE VOU FAZER DEPOIS

Obviamente esse conjunto de dados é extremamente simplista e serve apenas para fins de demonstração, mais tarde mostrarei um conjunto mais adequado para exemplos futuros.

Rodando uma regressão temos uma curva com o seguinte formato.

INSERIR GRÁFICO DA REGRESSÃO LINEAR SEM OUTLIER

Podemos avaliar então, se o valor da regressão for maior que 0,5 clasificamos o ponto como 1, que é provavél que seja maligno, caso contrário 0, benigno. Embora tal solução pareça eficiente ela não se sustenta caso estressemos um pouco montando um conjunto de dados um pouco diferente

CONJUNTO COM OUTLIER

No novo conjunto acima o ponto mais a esquerda é 1, só que seu valor é grande demais e a regressão linear vai acabar capturando o efeito desse ponto muito maior que os outros e mudando os parâmetros e formando a curva que segue. E, por consequência outros valores que claramente deveriam ser marcados como 1 acaba sendo mal classificados como 0.

INSERIR REGRESSÃO COM OUTLIER

Tal problema se torna ainda maior conforme a fronteira de decisão, isso é, quando algo é 0 ou 1, se torna ainda mais enevoada. Precisamos de um novo tipo de algoritmo para lidar com isso, problemas como esse, em que estamos tentando estimar um valor discreto (0 ou 1) chamamos de **problemas de classificação e, consequentemente, precisamos de algoritmos de classificação**.

Um dos algoritmos mais comuns para problemas de classificação é a regressão logística, ela é baseada na curva logística inicialmente criada no século 19 pelo matemático belga Pierre François Verhulst como um modelo de crescimento populacional. Não é claro porque ele a chamou de "logística". Anos depois, já no século 20 se observou que ela pode ser usada como uma fronteira de decisão quando queremos estimar a probabilidade de uma classe (por isso chamamos de regressão inclusive, mesmo sendo usado em problema de classificação) e como modelo se tornou amplamente utilizada em uma série de problemas.

INSERIR FOTO DA CURVA LOGÍSTICA PODE SER DA WIKIPEDIA 

Ao longo deste capítulo usarei como dados de exemplo uma pesquisa que catalogou a presença de uma espécie em extinção de aranha lobo em praia pela costa do Japão, assim como o tamanho dos grão de areia na praia em questão. O gráfico abaixo nos mostra esses dados, sendo o eixo y a presença ou não da aranha (1 ou 0) e o no eixo x o tamanho dos grãos de areia.

INSERIR DADOS

## Definição formal

Podemos definir a regressão logística como um algoritmo usado para classificação, ele pode ser usado tanto em problemas onde há múltiplas classes, por exemplo dada a foto de uma fruta identificar se é uma maça, banana, pera, etc; quanto binárias, 0 ou 1. Por enquanto vamos focar em problemas de classificação binária. Dizemos que o valor 1 é a presença daquilo que estamos estimando e 0 a não presença, um tumor ser maligno ou não, um email, ser spam ou não, etc.

Como no nosso caso estamos focando em valores binários não precisamos que nossa função atinja qualquer que não esteja dentro do intervalo $\{0,1\}$, aqui entra a função logística. Sua imagem é definida dentro do intervalo fechado $\{0,1\}$, a equação para sua construção é a que segue.

$$
y = \frac{1}{1+e^{-x}}
$$

Contudo, essa equação é "fechada", não aceita parâmetros se não o próprio $x$, logo como podemos alterar ela para se encaixar no nosso modelo.

O elemento da equação que controla a "forma" da equação, isso é, o quão para esquerda ou direita a curva está deslocada, ou o quão inclinada é a curva é o elemento $e^{-x}$, mais especificamente o seu expoente $-x$, logo alterando ele podemos criar outras curvas.

Essas outras curvas podem se encaixar ao nosso conjunto de dados, queremos uma que melhro represente os nossos dados de forma que, ao passarmos uma observação, a função nos retorna a probabilidade da observação pertencer a classe 1.

INSERIR EXEMPLO DE COMO PODEMOS MUDAR A CURVA USANDO PARAMETROS

Podemos então ter um conjunto de parâmetros $\theta$ e dados $x$, ambos representados como vetores, e colocá-los como nosso expoente. Dessa forma podemos construir um número de curvas epara ter aquela que melhor se adequa ao conjunto de dados que temos.

$$
h_{\theta}(x) = \frac{1}{1+e^{\theta^Tx}}
$$

Em outros problemas como regressão linear há um fórmula que nos dá o valor ideal, porém, com regressão logística não há nenhum algoritmo que nos dá os valores ideias. Existem casos especiais onde se sabe como achar uma solução analítica para o problema de otimização, mas de forma geral ainda não se descobriu um formato ideal de curva dado um conjunto de pontos.

Nos resta então a pergunta, de como podemos saber achar esse conjunto de $\theta$s dado que não temos uma solução analítica definida para o problema. Para isso temos que usar algoritmos de otimização, assim como pode ser feito com [regressão linear e descida de gradiente](regressao_linear.md#metodo-de-gradiente) (na verdade podemos usar o mesmo [método de gradiente](../../matematica/otimizadores/metodo_gradiente.md) e vamos explorar seu uso mais a frente). Contudo, antes de pularmos para os métodos vamos ver como podemos ter certeza de que há uma função que podemos otimizar e que podemos usar como função de custo para um algoritmo como descida de gradiente.

Vamos definir a função de probabilidade para cada uma das ocorrências da variável resposta $y$,

$$
\displaylines{
p(y=1|x;\theta)=h_{\theta}(x)\\
p(y=0|x;\theta)=1-h_{\theta}(x)
}
$$

A segunda equação é consequência da primeira, obviamente, se a probabilidade de ser da classe 1 é de 90%, a probabilidade de ser da classe 0 é de 10% (1-0,9).

Podemos reorganizar ambas equação em somente uma da seguinte forma,

$$
p(y|x;\theta)=(h_{\theta}(x))^y(1-h_{\theta}(x))^{1-y}
$$

Organizar a equação dessa forma é mais um "truque" do que qualquer outra coisa. O leitor pode verificar que em ambos os cenários, seja 0 ou 1, caso coloquemos o valor de $y$ na fórmula nos resta somente aquele que foi explicitado no par de equações anteriores.

Com isso, podemos calcular a verossimilhança de $p(\hat{y}|X;\theta)$, assumindo $m$ como um conjunto de dados em que cada ponto foi gerado de forma independente, temos que

$$
\begin{align}
\mathcal{L}(\theta)&=p(\hat(y)|X;\theta) \newline
&=\prod_{i=1}^{m}p(y^{(i)}|x^{(i)};\theta) \newline
&=\prod_{i=1}^{m}(h_{\theta}(x)^{(i)})^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}} \newline
\ln(\mathcal{L}(\theta))&=\ln\left(\prod_{i=1}^{m}(h_{\theta}(x)^{(i)})^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}\right) \newline
\ell(\theta)&=\sum_{i=1}^{m}y^{(i)}\ln(h_{\theta}(x^{(i)}))+(1-y^{(i)})\ln(1-h_{\theta}(x^{(i)}))
\end{align}
$$

Derivando o logaritmo da verossimilhança chegamos na equação abaixo,

$$
\frac{\partial \ell(\theta)}{\partial\theta} = \sum_{i=1}^{m} \left( y^{(i)} - h_{\theta}(x^{(i)}) \right)x^{(i)}
$$

Quem tem familiaridade com regressão linear vai notar que essa equação é exatamente igual a derivada da função de custo de uma regressão linear. O motivo para isso é porque ambos fazem parte de uma mesma família de algoritmos chamada de "modelos lineares generalizados", em outro capítulo falarei mais sobre esse tipo de algoritmo.

> [!NOTE] Derivação da verossimilhança
> Antes de qualquer coisa vamos definir a derivada da logística $g(z)$
> $$
> \begin{align}
> \frac{\partial g(z)}{\partial z} &= \frac{d}{dz} \frac{1}{1+e^{-z}} \newline
> &= \frac{1}{(1+e^{-z})^2}(e^{-z}) \newline
> &= \frac{1}{1+e^{-z}} \times \left( 1 - \frac{1}{1+e^{-z}} \right) \newline
> &= g(z)(1-g(z))
> \end{align}
> $$
> Essa derivação sera útil logo a frente, vamos então derivar $\ell(\theta)$ usando essa informação e notação,
> $$
> \begin{align}
> \frac{\partial \ell(\theta)}{\partial\theta} &= \frac{\partial}{\partial\theta}\sum_{i=1}^{m}y^{(i)}\ln(g(\theta^Tx^{(i)}))+(1-y^{(i)})\ln(1-g(\theta^Tx^{(i)})) \newline
> &= \sum_{i=1}^{m} \frac{\partial}{\partial\theta} \left(y^{(i)}\ln(g(\theta^Tx^{(i)}))+(1-y^{(i)})\ln(1-g(\theta^Tx^{(i)}))\right) \newline
> &= \sum_{i=1}^{m} \left( y^{(i)}\frac{1}{g(\theta^Tx^{(i)})}g(\theta^Tx^{(i)}) (1-g(\theta^Tx^{(i)}))x^{(i)} - (1-y)(\frac{1}{1-g(\theta^Tx^{(i)})})g(\theta^Tx^{(i)})(1-g(\theta^Tx^{(i)}))x^{(i)} \right) \newline
> &= \sum_{i=1}^{m} \left( y^{(i)}(1-g(\theta^Tx^{(i)}))x^{(i)} - (1-y)g(\theta^Tx^{(i)})x^{(i)} \right) \newline
> &= \sum_{i=1}^{m} \left( y^{(i)}(1-g(\theta^Tx^{(i)})) - (1-y)g(\theta^Tx^{(i)}) \right)x^{(i)} \newline
> &= \sum_{i=1}^{m} \left( y^{(i)}-g(\theta^Tx^{(i)})y^{(i)} - g(\theta^Tx^{(i)}) - g(\theta^Tx^{(i)})y^{(i)} \right)x^{(i)} \newline
> &= \sum_{i=1}^{m} \left( y^{(i)} - g(\theta^Tx^{(i)}) \right)x^{(i)} \newline
> \end{align}
> $$
> O único ponto que levanto a atenção é na terceira linha onde expando as derivadas e troco o sinal que separa as duas partes da equação, de mais por menos. Isso porque ao derivarmos $(1-g(\theta^{T}x^{(i)}))$ temos $-\frac{\partial g(\theta^{T}x^{(i)})}{\theta}$.

Com a derivada temos então a nossa função de custo para poder usar em algum método como gradiente ascendente ou método de newton para poder achar o melhor conjunto de $\theta$ dado um grupo de observações.

Nas seções abaixo vamos explorar o uso desses algoritmos para poder resolver um problema de regressão logística.

## Gradiente Ascendente

Quando trabalhamos com regressão linear (INCLUIR AQUI LINK DEPOIS) nós usamos o algoritmo de descida do gradiente (do inglês, *gradient descent*), $\theta_j := \theta_j - \alpha\frac{\partial\ell(\theta)}{\theta}$ (tecnicamente usei a função de custo $J(\theta)$, mas usar o logaritmo da verossimilhança é equivalente e, para fins de generalização, mais correto), mas aqui queremos lidar com o gradiente ascendente (do inglês, *gradient ascent*).

Ao termos a função de verossimilhança $\mathcal{L}(\theta)$, ou o seu logaritmo $\ell(\theta)$, queremos maximizar a probabilidade do conjunto de parâmetros $\theta$ se "encaixar" ao conjunto de dados $X$. Dessa forma o nosso otimizador quer maximizar o resultado e não minimizar. Para o método do gradiente, isso é dado invertendo o sinal do fator de ajuste do parâmetro $\theta_j$.

$$
\theta_j := \theta_j + \alpha\frac{\partial\ell(\theta)}{\theta}
$$

$\alpha$ ainda é a nossa taca de aprendizado, o quão o rápido ou devagar o nosso algoritmo converge para uma resposta. Aplicando o resultado da derivada da seção anterior temos então que a regra de atualização é:

$$
\theta_j := \theta_j + \alpha \left( \sum_{i=1}^m(y^{(i)} - h_{\theta}(x^{(i)}))x_j^{(i)} \right) \space \text{, para cada j}
$$

### Implementação em Python

Em construção.

### Implementação em Julia

Em construção.

## Método de Newton

Na seção anterior vimos o uso do método do gradiente para resolver um problema de regressão logística, contudo essa não é a aplicação mais comum. Pacotes como o SciKit-Learn em Python tem diversas opções de algoritmos para solucionar o problema, os chamados *solvers*, um dos mais comuns é o método de Newton. Resumindo-o em uma frase, ele funciona achando a tangente da curva que da processo que estamos tentando otimizar no ponto em que estamos atribuindo o nosso primeiro "chute" do parâmetro $\theta$, para em seguida usar onde essa tangente corta o eixo $x$ como valor de ajuste até chegarmos onde a nossa função original é igual a zero.

Podemos então pensar nele graficamente da seguinte forma

INSERIR GRÁFICO DA OTIMIZAÇÃO POR MÉTODO DE NEWTON

Vamos formalizar esse processo que descrevi acima.

O método de newton, assim como o método do gradiente, serve para podermos achar o mínimo local/global de alguma função $f$, isso acontece através de repetidos ajustes nos parâmetros da função que permitem que cheguemos o valor ótimo.

Queremos então achar um $\theta$ tal que $f(\theta)=0$. Dado um palpite inicial sobre o valor de $\theta$ podemos atualizar o valor de $\theta$ com

$$
\theta := \theta - \frac{f(\theta)}{f'(\theta)}
$$

Repetimos essa atualização até $\theta$ convergir.

Essa atualização nos traz exatamente o mesmo resultado que aquele que expressei graficamente antes!

Agora, o algoritmo que apresentei aqui serve para minimizarmos uma função, mas no caso da nosssa função de verossimilhança $\ell(\theta)$ queremos achar seu ponto máximo. Contudo, o máximo de $\ell$ é o local onde $\ell(\theta)=0$, então podemos construir o nosso fator de atualização como:

$$
\theta := \theta - \frac{\ell'(\theta)}{\ell''(\theta)}
$$

Até aqui pensamos em $\theta$ como um número real, mas no nosso problema de regressão logística ele é um vetor de tamanho $n+1$ (aqui incluo o intercepto, por isso somo um), assim temos que alterar o nosso fator de atualização para acomodar isso, felizmente isso é tão simples quanto.

$$
\theta := \theta - \nabla^2\ell(\theta)^{-1} \nabla\ell(\theta)
$$

Onde $\nabla$ é a derivação de um vetor, ou matriz (INSERIR LINK DEPOIS). A derivada de segunda ordem é um caso especial aqui, chamada de **matriz Hessiana** ($H$). Ela tem a forma de $\mathbb{R}^{n+1\times n+1}$ e seus elementos são dados por:

$$
H_{ij} = \frac{\partial^2\ell(\theta)}{\partial\theta_i \partial\theta_j}
$$

Uma propriedade interessante do método de Newton, e que permite que ele seja tão rápido em convergir à um valor, é que se chama de "convergência quadrática". Informalmente podemos dizer que, a cada iteração, o erro cai pelo quadrado do erro atual. Exemplo, se a distância entre o zero da função e o nosso parâmetro é de 0,01, na segunda iteração essa distância vai ser de 0,0001, na terceira 0,00000001...

Só que tal propriedade vem com um custo, conforme o número de dimensões de $\theta$ cresce, os número de parâmetros, se torna cada vez mais difícil computacionalmente resolver esse sistema de equações. Por exemplo, embora seja relativamente inverter uma matriz quadrada com 10 linhas e colunas, uma com 100,000 linhas é muito mais complexa.

### Implementação em Python

### Implementação em Julia

## Regressão multinomial (multiclasse)

Até agora vimos o caso em que temos somentes duas classes para a classificação, um email é spam ou não, a pessoa está doente ou não, etc. Mas e se tivermos mais de duas classes para se classificar? Um *dataset* já clássico em aprendizado de máquina é o Iris, que reúne informações sobre o comprimento e largura de das sépalas e pétalas de três diferentes espécies de de flores íris, como construiríamos um algoritmo para, dado os nossos *inputs* (tamanho e largura das partes da flor) qual é a espécie dessa flor? Para esse caso usamos uma forma especial da regresão logística, a **regressão *softmax***, ou **regressão logística multinomial**.

Suponhamos que temos um conjunto de dados $X$ com um certo de *features* e um *target* que pode assumir $K$ valores, temos então $K$ diferentes classes. Dado uma observação $x^(i)$ como podemos saber a que classe $k$ ele pertence? O método geral segue o mesmo, precisamos de uma função de custo para aplicarmos algum algoritmo de otimização como o método do gradiente ou método de Newton, para assim termos os $\theta$ ótimos para a classe. Aqui já temos a primeira diferença em relação a regressão logística binomial, no caso multinomial definimos uma matrix de parâmetros $\Theta$ de tamanho $\mathbb{R}^{K \times 1}$, onde cada linha é o vetor $\theta^{(k)}$ que contém os parâmetros para estimar a classe $k$ como se estivéssemos no caso binomial.

Antes disso, contudo, temos que redefinir a nossa função a ser estimada. Enquanto no caso binomial utilizávamos a função sigmoide em sua forma normal $\frac{1}{1+\exp{-x}}$, aqui usamos uma versão modificada dela, a chamada **função softmax**.

$$
\sigma(s(x))_k = \frac{\exp(s_k(\theta_k^T x^{(i)}))}{\sum_{j=1}^{K}\exp(s_j(\theta_j^T x^{(i)}))}
$$

Onde $K$ é o número de classes; $s_k(x^{(i)})$ é o escore da classe $k$ para a observação $x^{(i)}$, que pode é definido como $s_k(x^{(i)}) = x^{(i)}\theta_k^T$; e $\sigma(s(x))_k$ é probabilidade estimada de que $x^{(i)}$ pertença a classe $k$ considerando os escores de cada uma das possíveis classes, também podemos denotar isso como $\hat{p}_k$. 

No processo de estimação, calculamos a probabilidade de pertencer a cada classe $k$ e selecionamos a de maior como sendo a de maior valor.

Contudo, como podemos definir quais os melhores parâmetros para nossa equação? Como dito anteriormente, regressão logística, independente do tipo, não tem nenhuma solução fechaada, então temos de usar métodos como o do gradiente para chegar em uma solução. Para isso definimos então uma função e custo $J(\Theta)$ para melhor definir os parâmetros para cada $\theta$. Para isso usamos uma ferramenta que vem da teoria da informação, a **entropia cruzada**. (DEPOIS EU PENSO NO QUE ELA SIGNIFICA EXATAMENTE QUE EU NÃO ENTENDI)

Definimos então a função de custo $J(\Theta)$ como

$$
J(\Theta) = -\sum_{i=1}^{m} \sum_{k=1}^{K}y_k^{(i)}\ln(\hat{p}_k^{(i)})
$$

A soma sobre $k$, onde $K$ é o número de classes possíves, é a nossa entropia cruzada.

Podemos então derivar essa equação para chegar no fator de ajuste. Porém, caso se lembrem sobre o que foi comentado sobre modelos lineares, já sabem que essa função terá um formato específico mesmo sem cálculá-lo, $\sum_{i=1}^m(y^{(i)} - h_{\theta}(x^{(i)}))x_j^{(i)}$, assim pulamos por enquanto a derivação e podemos então já ter uma regra de ajuste para o nosso método de gradiente.

$$
\theta_k := \theta_k + \alpha \left( \sum_{i=1}^m (\hat{p}_k^{(i)} y^{(i)})x^{(i)} \right)
$$

## Outras implementações

Implementações como a do SciKit Learn trazem *solvers* adicionais para a regressão logística. Ao longo dessa seção vamos disutir esses dois outros métodos, o *liblinear* e o "Algoritmo de Memória Limitada de Broyden–Fletcher–Goldfarb–Shanno", ou *lbfgs*. Outros algoritmos implementados são baseados em descida de gradiente ou método de newton apresentado anteriormente.

### liblinear

*Liblinear* ou *Library for Large Linear Classification* (em português, Biblioteca para Classificações Lineares Extensas) é originalmente uma biblioteca em C, e a implementação do SciKit Learn depende da versão em C++ para o seu funcionamento. O algoritmo implementado é um de descida coordenada (do inglês, *coordinate descent*) que é relativamente similar a descida de gradiente com uma peculiaridade

---

## Referências

**Stanford Online** (2020). Locally Weighted & Logistic Regression | Stanford CS229: Machine Learning - Lecture 3 (Autumn 2018). https://www.youtube.com/watch?v=het9HFqo1TQ.

https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions

**Aurélien Géron**. Mãos à Obra: Aprendizado de Máquina com Sciki-Learn, Keras & TensorFlow. 2ª edição.

https://youtu.be/9qFABxUQTrI?si=sd34dgLg-FdgcEDf

https://www.youtube.com/watch?v=TiiF3VG_ViU