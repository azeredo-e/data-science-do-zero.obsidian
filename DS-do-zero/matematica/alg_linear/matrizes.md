# Matrizes

## Operações com matrizes

Ainda em expansão, mas vou deixar aqui algumas definições que sobre derivadas de matrizes e outras propriedades.

### Notação $\nabla$ (nabla) para derivadas

Uma notação que é particularmente útil quando estamos lidando com derivadas de matrizes é a *nabla* $\nabla$. Ao longo das notas ela deve aparecer as vezes então deixo aqui uma explicação completa de como interpretar essa notação.

Para uma função $f:\mathbb{R}^{m \times n} \to \mathbb{R}$, ou seja uma que recebe uma matriz de tamanho $m$ por $n$ e retorna um número real, podemos definir a derivada de $f$ em respeito a $A$, uma matriz, como sendo:

$$
\nabla_Af(A) = 
  \left[ {\begin{array}{ccc}
    \frac{\partial f}{\partial A_{11}} & \dots & \frac{\partial f}{\partial A_{1n}} \\
    \vdots & \ddots & \vdots \\
    \frac{\partial f}{\partial A_{m1}} & \dots & \frac{\partial f}{\partial A_{mn}} \\
  \end{array} } \right]
$$

Podemos ler $\nabla_Af(A)$ como "a derivada de $f$ em respeito a $A$". Por exemplo, temos uma matriz $2\times2$ definida como:

$$
A =
    \left[ {\begin{array}{cc}
        a_{11} & a_{12} \\
        a_{21} & a_{22} \\
    \end{array} } \right]
$$

E uma função $f$ definida como $f(A) = \frac{3}{2}A_{11} + 5A_{12}^2+A_{21}A_{22}$. Então derivada de $f$ em respeito a $A$ seria:

$$
\nabla_Af(A) = 
    \left[ {\begin{array}{cc}
        \frac{3}{2} & 10A_{12} \\
        A_{22} & A_{21} \\
    \end{array} } \right]
$$

Uma propriedade dessa notação é a seguinte:

$$\nabla_A\det(A)=\det(A)(A^{-1})^T$$
Sendo $\det$ o determinante da matriz $A$.

### Traço

O operador "traço" (do inglês, *trace*), $tr$, pode ser interpretado como uma função que mapeia $\text{tr}:\mathbb{R}^{n \times n} \to \mathbb{R}$ e é definido da seguinte forma:

$$
\text{tr}(A_{n \times n}) = \sum^n_{i=1}A_{ii}
$$

Isso é, o $trA$ (lê-se o traço de $A$, sendo $A$ uma matriz quadrada) é a soma dos itens em sua diagonal principal.

Segue aqui algumas propriedades desse operador. (A PROVA EU TENHO QUE PENSAR DEPOIS)

$$\text{tr}(A) = \text{tr}(A^T)$$
*Isso é, uma matriz e sua transposta tem o mesmo traço.*

$$\text{tr}(A+B)=\text{tr}(A)+\text{tr}(B)$$

$$\text{tr}(kA)=k\text{tr}(A)$$
*Sendo $k$ um escalar qualquer.*

$$\nabla_A \text{tr}(AB) = B^T$$

$$\nabla_{A^T}f(A)=(\nabla_Af(A))^T$$

$$\nabla_A \text{tr}(ABA^TC)=CAB+C^TAB^T$$
