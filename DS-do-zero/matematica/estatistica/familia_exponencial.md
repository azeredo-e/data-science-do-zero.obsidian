# Família Exponencial

A família exponencial é todo um conjunto de distribuições estatísticas que seguem um formato específico. Uma distribuição pode ser pertencente a família exponencial caso a sua definição possa ser escrita na forma

$$
\tag{1}
p(y;\eta) = b(y)\exp(\eta^T T(y) - a(\eta))
$$

Um detalhe que salta aos olhos logo de cara ao ver essa equação é os próprios argumentos que ela recebe, $y$ parametrizado por $\eta$ (lê-se "eta"), por que estamos usando a letra comumente utilizada para o resultado da função como sua variável? A resposta para isso é porque aqui estamos interessados na distribuição da variável resposta $y$, não naquilo que gera ela, ou seja, dado uma função $f(x)=y$ qual a $p(y;\eta)$.

Os componentes da equação são então: $y$, que é o nosso próprio conjunto de dados; $\eta$, chamado também de parâmetro natural, é um parâmetro subjacente a própria distribuição dos dados; $T(y)$ é chamada de estatística suficiente (em termos simples, se diz que uma estatística é suficiente, se dado o valor de $T(y)$ podemos ter uma estimativa tão boa sobre aquele que parametriza a variável $y$, $\eta$, sabendo só o valor dessa estatística do que sabendo todos os valores da amostra), aqui, para a maioria dos casos, será o próprio $y$; $b(y)$ é o parâmetro base EXPANDIR DEPOIS; e, por fim, $a(\eta)$ é função de partição logarítmica, de maneira prática ele normaliza a expressão e garante que a integral definida de toda a curva seja igual a 1.

Um segundo ponto que chama a atenção é que na equação uso $\eta^T$, isso porque como o $\eta$ é um parâmetro dos nossos dados $y$, no caso deles serem também um vetor (esse caso pode ser visto quando aplicamos GLM a uma regressão *softmax*) temos que usar $\eta^T$ pois este é um vetor assim como $T(y)$, que aqui teria as mesmas dimensões de $\eta^T$. Mesmo assim, no final o valor final do problema é um escalar visto que todos os outros componentes são sempre escalares.

Por fim, fazemos aqui uma afirmação então, para um dado conjunto de $a(y)$, $b(y)$ e $T(y)$, caso aplicados a função acima e estes integrem $y$ à 1, o resultado é a função densidade de probabilidade (FDP) de alguma família de distribuições e tal família pertence a família exponencial. Por exemplo, dado uma certa escolha dos nossos componentes podemos ter a família de distribuições Gaussinas, e a escolha do $\eta$ determina exatamente qual membro desta famĺia que estamos lidando com.

## Propriedades

Há uma série de propriedades que seguem da família exponencial. Vamos aqui mostrar algumas das mais relevantes

**Funções estritamente côncavas**

Para uma função pertencente a família exponencial, caso se aplique MLE (*maximun likelihood estimation*) para se achar os parâmetros $\eta$ da distribuição, o resultado do problema de otmização é estritamente côncavo, ou seja, ao aplicar um método como descida de gradiente, ele não ficará preso em mínimos locais e tenderá ao máximo global.

De forma inversa, o ao usar o log da verossimilhança, o problema de otimização resultante é estritamente convexo.

**Esperança e Variância**

$E(y;\eta) = \frac{\partial a(\eta)}{\partial \eta}$ e $Var(y;\eta) = \frac{\partial^2 a(\eta)}{\partial \eta^2}$ o que é extremamente útil já que não precisamos integrar a FDP para acharmos tais valores.

## Famílias de distribuições

Ao longo das próximas sessões irei demonstrar como que algumas famílias de distribuições típicas da estatística pertencem a família exponencial.

### Distribuição de Bernoulli

A distribuição de Bernoulli descreve uma distribuição discreta dentro do espaço amostral $y \in \{0,1\}$ de tal forma que $p(y=1;\phi) = \phi$ e $p(y=0;\phi) = 1-\phi$, sendo $\phi$, lê-se "fi", a média. Por variar $\phi$ obtemos diferentes distribuições dentro da família Bernoulli. Um exemplo de evento que segue uma distribuição de Bernoulli é o jogar de uma moeda, podemos obter somente cara ou coroa.

Podemos escrever a probabilidade de $y$ dado $\phi$ dessa distribuição como:

$$
p(y;\phi) = \phi^{y}(1-\phi)^{1-y}
$$

Dado essa fórmula podemos rearranjar ela como

$$
\begin{align}
p(y;\phi) &= \phi^{y}(1-\phi)^{1-y} \newline
&=\exp(\log(\phi^{y}(1-\phi)^{1-y})) \newline
&=\exp(y\log(\phi) + (1-\phi)\log(1-\phi)) \newline
\tag{2}
&= \exp\left( \log\left(\frac{\phi}{1-\phi}\right)y + \log(1-\phi) \right)
\end{align}
$$

Com essa nova expressão podemos organizar os nossos componentes da família exponencial padrão. Temos então que

$$
\begin{align}
b(y) &= 1 \newline
\eta &= \log\left( \frac{\phi}{1-\phi} \right) \implies \phi=\frac{1}{1-e^{-\eta}} \newline
T(y) &= y \newline
a(\eta) &= - \log(1-\phi) \newline
&= \log(1+e^{\eta})
\end{align}
$$

Levanto um ponto de atenção aqui. Para $a(y)$ troco o sinal de como aparece na equação $2$ pois, como pode ser visto na equação $1$, usamos $-a(y)$.

> [!NOTE] Sobre o valor de $\phi$
> O olhar atento vai notar que $\phi$ é a função sigmoide. Isso se mostrará relevante quando se estudar GLM e formos ver sobre a regressão logística, mas desde já digo que só por isso podemos concluir que a escolha da função sigmoide não é a toa quando se trata da regressão logística.

Com isso demonstramos que a FDP de Bernoulli pode ser escrita no formato da equação $1$ e, por consequência de ser uma FDP, ela integra à 1. Com ambos esses pontos demonstrados podemos então concluir que a distribuição de Bernoulli faz parte da família exponencial!

### Distribuição Gaussiana

A distribuição Gaussiana, também chamada de distribuição normal, é um tipo de distribuição estatística contínua muito usada nos mais diferentes ramos da ciência. Aqui veremos como ela, também, faz parte da família exponencial.

Pensando em como podemos usar a família exponencial no contexto de GLMs podemos simplificar a PDF da Gaussiana pois a variância $\sigma^2$ não altera o resultado. Logo, temos a PDF

$$
p(y;\mu) = \frac{1}{\sqrt{2\pi}}\exp\left( -\frac{(y-\mu^2)}{2} \right)
$$

Que pode ser reescrita como

$$
p(y;\mu) = \frac{1}{\sqrt{2\pi}}\exp\left( -\frac{1}{2}y^2 \right) \cdot\exp\left( \mu y - \frac{1}{2}\mu^2 \right)
$$

Que pode ser mapeada para os atributos da família exponencial como

$$
\begin{align}
b(y) &= \frac{1}{\sqrt{2\pi}}\exp\left( -\frac{1}{2}y^2 \right) \newline
\eta &= \mu \newline
T(y) &= y \newline
a(\eta) &= \frac{\mu^2}{2} \newline
&= \frac{\eta^2}{2}
\end{align}
$$

### Distribuição Multinomial

### Distribuição de Poisson

### Distribuição Gamma

### Distribuição Geométrica

### Distribuição Wald (Inverso Gaussiana)

### Distribuição Binomial Negativa

## Referências

**Stanford Online** (2020). Lecture 4 - Perceptron & Generalized Linear Model | Stanford CS229: Machine Learning - Lecture 3 (Autumn 2018). https://youtu.be/iZTeva0WSTQ?si=6BMmlNMIR2_DLJik.

https://math.arizona.edu/~tgk/466/sufficient.pdf
