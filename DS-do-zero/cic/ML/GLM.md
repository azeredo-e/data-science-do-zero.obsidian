# Modelos Lineares Generalizados

A ideia por trás dos modelos lineares generalizados (MLG, mas muitas vezes escrito como GLM por causa do seu nome em inglês) é que, caso saibamos a distribuição estatística da nossa variável resposta, podemos achar um um modelo de regressão para esta que consegue se encaixar aos nossos dados. Queremos estimar quantos clientes vão visitar uma loja, isso é um problema clássico para uma distribuição Poisson; ou no caso de estimar eventos em que o resultado de um não influência o outro num tempo contínuo, este é um clássico exemplo de um um problema de distribuição normal; entre outros.

MLGs nos atendem em poder resolver toda essa gama de problemas. De forma geral, dado que queremos prever uma dada variável aleatória $y$ como um função de $x$, podemos usar o método de MLG para construir um estimador $\hat{y}$ para tal $y$. Para tal assumimos que: 

1. $y | x;\theta \sim ExponentialFamily(\eta)$;
2. Dado um $x$, queremos estimar o valor esperado de $T(y)$ dado $x$. Como na maioria dos casos $T(y)=y$ queremos então que a nossa hipótese de predição $h$ satisfaça $h(x)=E(y|x)$;
3. O parâmetro natural $\eta$ e $x$ estão relacionados linearmente: $\eta=\theta^Tx$.

A terceira hiótese é menos intuitiva dessas, e, em vários sentidos é menos uma hipótese e mais uma escolha sobre como queremos que o nosso modelo se comporte.

Nas próximas sessões será demonstrado como criar uma modelo do tipo MLG através de dois exemplos, regresão linear e logística, após isso uma definição formal sobre a construção será dada.

### Construindo MLGs

Uma nota histórica de importância aqui é que: as derivações expostas aqui podem parecer como a forma como esses modelos foram desenvolvidos. O caso é na verdade é o contrário, muitos modelos já existiam, e um paper escrito por John Nelder e Robert Wedderburn chamado "Generalized Linear Models" (1972) onde descrevem como esses modelos podem ser unificados sob um único *framework*, mais detalhes sobre isso aparecerão quando discutirmos sua definição formal.

### Para o caso de Mínimo Quadrados Ordinários

Como visto quando estudado regressão linear, supomos que a distribuição da nossa variável *target* segue uma distribuição normal, isso é $y \sim \mathcal{N}(\mu, \sigma^2)$, esta que faz parte da família exponencial. A distribuição pode ser escrita como membro da família exponencial como $p(y;\mu) = \frac{1}{\sqrt{2\pi}}\exp\left( -\frac{1}{2}y^2 \right) \cdot\exp\left( \mu y - \frac{1}{2}\mu^2 \right)$ o que nos dá que $a(\eta)=\frac{\eta^2}{2}$.

Olhando para as nossas escolhas de design sobre os MLG vemos que, segundo a suposição 2, temos que $h(x)=E(y|x)$, e, como visto quando discutido a famĺia exponencial sabemos que $E(y;\eta) = \frac{\partial a(\eta)}{\partial \eta}$, logo $E(y;\eta)=\eta$, dessa forma temos que $h(x) = \eta$. Só que segundo a suposição 3, temos que $\eta=\theta^Tx$. Por fim temos que $h(x)=\theta^Tx$, que é exatamente a nossa suposição sobre regressão linear! Uma reta traçada sobre os nossos dados.

Tendo já uma função de hipótese podemos então usar algum método de otimização como descida de gradiente, ou Mínimos Quadrados Iterativamente Reponderado (em inglês IRLS) como veremos mais adiante, para ter os estimadores $\theta$.

### O caso logístico

No caso da regressão logística temos que estamos modelando dados binários, 0 ou 1, aplicando a distribuições conhecidas temos a distribuição de Bernoulli, $y \sim Bern(\phi)$, que faz parte da família exponencial. Esta pode ser escrito da seguinte forma $\exp\left( \log\left(\frac{\phi}{1-\phi}\right)y + \log(1-\phi) \right)$.

Com base nas suposições sobre a construção de um MLG temos que $h(x)=E(y|x)$, e, dado que estamos tratando da distribuição Bernoulli, temos que $E(y;\phi)=\phi$, só que $\phi=\frac{1}{1-e^{-\eta}}$ e, segundo a suposição 3, temos que $\eta=\theta^Tx$, logo, segundo a suposição 2 podemos definir que $h(x)=\frac{1}{1-e^{-\theta^Tx}}$, exatamente a função logística!

### Definição formal

Em construção.

## Mínimos Quadrados Iterativamente Reponderado (IRLS)

Em construção.

## Referências

**Stanford Online** (2020). Lecture 4 - Perceptron & Generalized Linear Model | Stanford CS229: Machine Learning - Lecture 3 (Autumn 2018). https://youtu.be/iZTeva0WSTQ?si=6BMmlNMIR2_DLJik.
