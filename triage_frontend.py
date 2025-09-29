# Triage - Gradio Interface Config

# Importação das bibliotecas principais
import gradio as gr                     # Criação da interface web
import torch                            # Inferência do modelo em PyTorch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import pymupdf                          # Leitura de arquivos PDF

# Carrega modelo e tokenizer do Hugging Face
# Esse modelo foi treinado realizando um fine-tuning no modelo
# DistilBERT base multilingual (cased), para especializa-lo
# na classificação de emails.
model_name = "gabubits/triage-portuguese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Labels do classificador
label_names = ["Improdutivo", "Produtivo"]

# Função de predição principal
def predict_email(assunto, conteudo, arquivo):

    # Evita que usuário insira texto e arquivo ao mesmo tempo
    if (assunto.strip() or conteudo.strip()) and arquivo is not None:
      return "## ❌ Erro: Por favor, faça somente o upload do email ou a descrição do e-mail"

    # Caso seja enviado arquivo .txt ou .pdf
    if arquivo is not None:
        if arquivo.name.endswith(".txt"):
            with open(arquivo.name, "r", encoding="utf-8") as f:
              conteudo = f.read()
        if arquivo.name.endswith(".pdf"):
            conteudo = "" 
            doc = pymupdf.open(arquivo.name)
            for page in doc:
              conteudo += page.get_text()

    # Exige que pelo menos o corpo do email seja enviado
    if not conteudo.strip():
        return "## ❌ Erro: Por favor, digite, pelo menos, o corpo do email"
    
    # Monta entrada para o modelo
    input_text = f"SUBJECT: {assunto} BODY: {conteudo}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    # Predição com PyTorch sem gradiente
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Probabilidades normalizadas
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().numpy()
    predicted_class = np.argmax(probs)
    
    # Emojis para facilitar visualização
    emoji_map = {
        "Produtivo": "✅",
        "Improdutivo": "❌"
    }
    
    # Montagem da resposta formatada
    result_label = label_names[predicted_class]
    emoji = emoji_map.get(result_label, "")
    result = f"## Resultado: {emoji} {result_label.upper()}\n\n"

    # Inclui assunto e conteúdo recebidos
    if assunto.strip():
        result += f"- **Assunto recebido:** {assunto}\n\n"
    else:
        result += f"- **Assunto recebido:** Não foi enviado um assunto.\n\n"
    result += f'- **Conteúdo recebido:** \n\n"{conteudo}"\n\n'

    # Sugestões automáticas de resposta
    responses = {
        "Produtivo": "Obrigado pelo contato.\nEm breve a sua mensagem será respondida. Aguarde!\n\nAtenciosamente, Fulano de Tal.",
        "Improdutivo": "Olá! A sua mensagem foi recebida. Muito obrigado!"
    }

    result += f"### Sugestão de resposta:\n{responses[result_label]}\n\n"
        
    return result

# Construção da interface com Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Triage - Classificador de e-mails")
    gr.Markdown("### Olá, esse é o Triage: um classificador de e-mails produtivos e improdutivos. O modelo principal foi refinado (fine-tuning) com base no modelo DistilBERT base multilingual (cased) e é especializado na classificação (produtivo ou improdutivo) de e-mails ou arquivos.")

    with gr.Row():
        # Coluna esquerda: inputs
        with gr.Column(scale=1):
            assunto = gr.Textbox(label="Assunto do e-mail", placeholder="Digite o assunto do e-mail")
            conteudo = gr.Textbox(label="Conteúdo do e-mail", placeholder="Digite o conteúdo do e-mail", lines=5)
            arquivo = gr.File(label="Anexar arquivo (.txt ou .pdf)", file_types=[".txt", ".pdf"])
            btn = gr.Button("Classificar", variant="primary")
        
        # Coluna direita: saída formatada
        with gr.Column(scale=1):
            saida = gr.Markdown()
    with gr.Row():
        gr.Examples(
                examples=[
                    ["Pedido de cópia do aditivo contratual de aumento salarial (Ref. 2024).", "Olá RH, Preciso de uma cópia do meu aditivo contratual que formaliza o último aumento salarial (Referência 2024) para fins pessoais. Por favor, envie o documento. Obrigado.", None],
                    ["Confirmação de Inscrição: Programa de Mentoria 2025.", "Olá, Sua inscrição no Programa de Mentoria 2025 foi confirmada com sucesso.", None],
                    ["Pedido de cotação para serviço de manutenção de ar-condicionado (Anual).", "Olá Compras, Precisamos de uma cotação para o serviço de manutenção anual preventiva e corretiva de todos os aparelhos de ar-condicionado do escritório. Por favor, solicite 3 propostas. Obrigado.", None],
                    ["Agradecimento pelo elogio ao Atendimento.", "Olá Fulano, Recebemos seu feedback positivo sobre o atendimento do nosso suporte. Repassamos o elogio à equipe.", None],
                ],
                inputs=[assunto, conteudo, arquivo]
            )

    # Configuração do botão (fica bloqueado durante execução + spinner de progresso)
    btn.click(
        fn=predict_email, 
        inputs=[assunto, conteudo, arquivo], 
        outputs=saida, 
        api_name="predict", 
        show_progress="full"
    )

# Inicializa aplicação
demo.launch(share=True)
