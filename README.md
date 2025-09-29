# Triage (Projeto desenvolvido para o Case Prático da empresa AutoU)

O **Triage** é um projeto feito como solução para o desafio proposto como Case Prático da empresa AutoU.

## Descrição breve do desafio

O objetivo central do case é **automatizar a leitura** e **sugerir classificações e respostas automáticas** de acordo com o teor de cada email recebido.

Existem duas categorias de classificação:

- Emails **Produtivos**: Emails que requerem uma ação ou resposta específica (ex.: solicitações de suporte técnico, atualização sobre casos em aberto, dúvidas sobre o sistema); e

- Emails **Improdutivos**: Emails que não necessitam de uma ação imediata (ex.: mensagens de felicitações, agradecimentos).

Para elevar a experiência de uso do usuário e do modelo, também foi desenvolvido uma **interface** com o **Gradio**, framework essencial para construção visual de modelos de machine learning.

Em síntese, o desafio é composto de três principais etapas:

1. Treinamento do modelo especializado em classificação de e-mails em produtivo/improdutivo em Python;
2. Criação da interface para o usuário classificar seus e-mails; e
3. Deploy da aplicação na nuvem.

## Solução

Para atigir esses objetivos, proponho o **Triage**, um sistema especialmente criado para esse case prático com foco na classificação de emails e resposta automática. Em resumo, o modelo principal do Triage foi treinado através de um
processo de _fine-tuning_ no modelo popularmente conhecido [DistilBERT base multilingual](https://huggingface.co/distilbert/distilbert-base-multilingual-cased) para especializa-lo na tarefa principal de classíficação de e-mails.

Para verificar o modelo treinado, ele foi exportado no formato Hugging Face Transformes e está disponível aqui: [https://huggingface.co/gabubits/triage-portuguese](https://huggingface.co/gabubits/triage-portuguese)

Para testar a aplicação, ela está deployada e hospedada no Hugging Face Spaces: [https://huggingface.co/spaces/gabubits/triage-demo-email-classifier](https://huggingface.co/spaces/gabubits/triage-demo-email-classifier).

## Arquivos principais

- [`dataset.csv`](dataset.csv) - Arquivo com a especificação do dataset usado para treinamento. Esse detaset é um conjunto de 391 e-mails classificados entre produtivo e improdutivo. Foi gerado de forma sintética e orientada com o apoio do Gemini.
- [`triage_model_training_process.ipynb`](triage_model_training_process.ipynb) - Arquivo com a especificação e detalhamento de todo o processo de treinamento do modelo DistilBERT base multilingual para especializa-lo em ser um classificador de e-mails.
- [`triage_frontend.py`](triage_frontend.py) - Arquivo de interface baseada em Gradio. Nesse arquivo é especificado o uso do modelo diretamente do Hugging Faces Transformers, o processo de predição de um novo e-mail e o design da interface, focado na simplicidade e nas funcionalidades.

**Observação**: O GitHub não permite arquivos maiores que 25MB, portanto todos os arquivos relacionados ao modelo treinado estão na página no Hugging Face Tranformers.
