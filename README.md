# Выразительная сила рекуррентных нейронных сетей / Expressive Power of Recurrent Neural Networks, II

**Автор:** Алкин Эмиль Венерович.

**Научный руководитель:** Оселедец Иван Валерьевич, доктор физико-математических наук.

## Постановка задачи
Один из новых методов изучения явления *depth power* в глубоких нейронных сетях (DNNs) представляет собой изучение скорости роста CP-ранга тензора $A$ при варьировании гиперпараметров архитектур нейронных сетей, чья score функция выражается в виде

$$h_y(x_1, \ldots , x_d ) = \sum_{i_1, \ldots, i_d = 1}^n A^y_{i_1, \ldots ,i_d} \prod_{j=1}^d f_{\theta_{i_j}}(x_i),$$

где $A^y_{i_1, \ldots ,i_d}$ - тензор обучаемых параметров, $f_{\theta_{i_j}}(x_i)$ - вектора признаков и $h_y(x_1, \ldots , x_d )$ - score функция для класса $y$.

Целью работы является теоретический анализ скорости роста ''типичного'' CP-ранга тензора обучаемых параметров, соответствующего рекуррентной нейронной сети, у которой внутренные слои имеют одинаковые параметры, при увеличении количества осей $d$ тензора $A$.

Целями экспериментов являются:

- численное подтверждение полученной нижней оценки на CP-ранг случайного тензора, у которого Tensor-Train ранг не больше заданного числа;
- сравнение эффективности рекуррентных нейронных сетей с разными и одинаковыми внутренними слоями на сложных датасетах.

## Основной теоретический результат

В файле ``tensors_strong_hyp_sketch.pdf`` изложено доказательство основного результата, сформулированного в виде теоремы:

**Theorem**

Suppose that $d = 2k$ is even. 
Define the following set 
```math
B := \left\lbrace \mathcal{X} \in \mathcal{M}^{eq}_{\mathbf{n}, \mathbf{r}} : \text{rank}_{CP} \mathcal{X} < q^{\frac{d}{2}} \right\rbrace,
```
where  $q = \text{min} \left\lbrace n, r-1 \right\rbrace$.
Then $\mu(B) = 0,$ where $\mu$ is the standard Lebesgue measure on $\mathcal{M}^{eq}_{\mathbf{n}, \mathbf{r}}$.

## Эксперименты

Эксперименты на датасете MNIST можно посторить, запустив код из Jupyter notebook`ов в папке ``exps_on_mnist``.

Численное подтверждение полученной нижней оценки на CP-ранг случайного тензора, у которого Tensor-Train ранг не больше заданного числа находится в Jupyter notebook`е ``estimate_cp_rank_for_random_tensor.ipynb``.

