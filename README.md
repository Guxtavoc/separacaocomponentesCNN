# CNN para Classificação de Componentes Eletrônicos

Projeto desenvolvido utilizando Redes Neurais Convolucionais (CNN) para classificação de componentes eletrônicos a partir de imagens.

O sistema é capaz de identificar:
- Capacitores
- Resistores
- Transistores
- Circuitos Integrados (IC)

Além do treinamento da CNN, foi desenvolvida uma aplicação web utilizando Flask para upload e classificação de imagens em tempo real.

---

# Sobre a escolha da CNN

A arquitetura CNN foi escolhida por sua alta eficiência em tarefas de visão computacional e reconhecimento de padrões visuais.

Diferentemente de métodos tradicionais de Machine Learning, as CNNs conseguem extrair automaticamente características relevantes das imagens, como:
- bordas
- formatos
- texturas
- padrões espaciais

Isso torna a abordagem ideal para classificação de componentes eletrônicos, que frequentemente possuem diferenças visuais sutis entre si.

---

# Arquitetura da Rede

```python
Sequential([
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(4, activation='softmax')
])
```
# Resultados

O modelo alcançou aproximadamente:

- 92% de acurácia
- Excelente desempenho em precision, recall e F1-score
- Boa capacidade de generalização para imagens não vistas

# Tecnologias Utilizadas
- Python
- TensorFlow / Keras
- Flask
- Docker
- HTML

# Dataset

Dataset utilizado:
[Basic Electronic Components](https://www.kaggle.com/datasets/julioazancort/basic-electronic-components)

# Treinamento no Google Colab

O treinamento do modelo foi realizado utilizando Google Colab devido ao suporte a GPU.

Notebook utilizado:
[https://colab.research.google.com/](https://colab.research.google.com/drive/1lxu8fkx9mdch8rqOTtrLRryUXVjSiPfT#scrollTo=PDL6_MtMHIAa)

# Executando com Docker
Build da imagem
```bash
docker build -t projetoia .
```
Executar container
```bash
docker run -p 5000:5000 projetoia
```

# Acessando a aplicação

Após iniciar o container:
```
http://localhost:5000
```
