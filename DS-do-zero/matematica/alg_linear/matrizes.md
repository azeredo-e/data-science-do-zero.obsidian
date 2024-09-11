# Operações com matrizes

Ainda em expansão, mas vou deixar aqui algumas definições que sobre derivadas de matrizes e outras propriedades.

## Notação $\nabla$ (nabla) para derivadas

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

## Traço

O operador "traço" (do inglês, *trace*), $tr$, pode ser interpretado como uma função que mapeia $tr:\mathbb{R}^{n \times n} \to \mathbb{R}$ e é definido da seguinte forma:

$$
trA_{n \times n} = \sum^n_{i=1}A_{ii}
$$

Embora o 



