import streamlit as st
import pandas as pd
import joblib
from scipy.stats.mstats import winsorize

# Função para aplicar winsorização
def apply_winsorization(data, limits):
    data_winsorized = data.copy()
    for column, limit in zip(data.columns, limits):
        data_winsorized.loc[:, column] = winsorize(data_winsorized[column], limits=limit)
    return data_winsorized

# Carregar os parâmetros salvos e o modelo para novos dados
try:
    scaler_X = joblib.load('pipeline_projeto_youtube/scaler_X.pkl')
    scaler_y = joblib.load('pipeline_projeto_youtube/scaler_y.pkl')
    modelo_GBR_v2 = joblib.load('pipeline_projeto_youtube/modelo_XGB_v2.pkl')
    winsor_limits = joblib.load('pipeline_projeto_youtube/winsor_limits.pkl')
    label_encoders = joblib.load('pipeline_projeto_youtube/label_encoders.pkl')
    X_treino_columns = joblib.load('pipeline_projeto_youtube/X_treino_columns.pkl')
except Exception as e:
    st.error(f"Erro ao carregar arquivos: {e}")

# Carregar os dados de mapeamento salvos
dados_ml = pd.read_csv('pipeline_projeto_youtube/dados_mapeamento.csv')

# Obter lista de países disponíveis e ordená-los alfabeticamente
paises = sorted(dados_ml['pais'].unique().tolist())

# Função para obter valores automáticos
def obter_valores_automaticos(pais):
    valores_pais = dados_ml[dados_ml['pais'] == pais].iloc[0]
    populacao_2022 = valores_pais['populacao_2022']
    total_usuarios_estimado_por_pais = valores_pais['total_usuarios_estimado_por_pais']
    return populacao_2022, total_usuarios_estimado_por_pais

# Função para calcular variáveis derivadas
def calcular_variaveis_derivadas(total_de_inscritos, total_de_visualizacoes, idade_do_canal,
                                 total_de_visualizacoes_ultimos_30_dias, total_de_inscritos_ultimos_30_dias,
                                 populacao_2022, total_usuarios_estimado_por_pais):
    proporcao_de_visualizacoes_recentes = total_de_visualizacoes_ultimos_30_dias / total_de_visualizacoes if total_de_visualizacoes != 0 else 0
    media_crescimento_inscritos_mes = total_de_inscritos_ultimos_30_dias / (idade_do_canal * 12) if idade_do_canal != 0 else 0
    return proporcao_de_visualizacoes_recentes, media_crescimento_inscritos_mes

# Função para preparar os dados para o modelo
def preparar_dados_para_modelo(total_de_inscritos, total_de_visualizacoes, pais, idade_do_canal,
                               total_de_visualizacoes_ultimos_30_dias, total_de_inscritos_ultimos_30_dias):
    # Obter valores automáticos
    populacao_2022, total_usuarios_estimado_por_pais = obter_valores_automaticos(pais)
    # Calcular variáveis derivadas
    proporcao_de_visualizacoes_recentes, media_crescimento_inscritos_mes = calcular_variaveis_derivadas(
        total_de_inscritos, total_de_visualizacoes, idade_do_canal, total_de_visualizacoes_ultimos_30_dias,
        total_de_inscritos_ultimos_30_dias, populacao_2022, total_usuarios_estimado_por_pais
    )
    # Criar DataFrame com os novos dados
    novos_dados = pd.DataFrame([[
        total_de_inscritos, total_de_visualizacoes, pais, idade_do_canal,
        total_de_visualizacoes_ultimos_30_dias, total_de_inscritos_ultimos_30_dias,
        total_usuarios_estimado_por_pais, populacao_2022,
        proporcao_de_visualizacoes_recentes, media_crescimento_inscritos_mes
    ]], columns=X_treino_columns)
    # Aplicar Label Encoding para a variável categórica 'pais'
    novos_dados['pais'] = label_encoders['pais'].transform(novos_dados['pais'])
    # Aplicar Winsorização
    variables_to_winsorize = novos_dados.columns.difference(['idade_do_canal'])
    novos_dados[variables_to_winsorize] = apply_winsorization(novos_dados[variables_to_winsorize], winsor_limits)
    # Separar colunas não escaladas
    non_scaled_columns = ['idade_do_canal']
    novos_dados_non_scaled = novos_dados[non_scaled_columns]
    novos_dados_to_scale = novos_dados.drop(columns=non_scaled_columns)
    # Aplicar padronização
    novos_dados_scaled = scaler_X.transform(novos_dados_to_scale)
    novos_dados_scaled = pd.DataFrame(novos_dados_scaled, columns=novos_dados_to_scale.columns, index=novos_dados.index)
    # Reconstruir DataFrame final
    novos_dados_final = pd.concat([novos_dados_scaled, novos_dados_non_scaled], axis=1)
    # Reordenar as colunas para garantir que estejam na mesma ordem que durante o treinamento
    novos_dados_final = novos_dados_final[X_treino_columns]
    return novos_dados_final

# Função para prever os ganhos mensais estimados
def prever_ganhos_mensais(novos_dados_final):
    # Prever usando o modelo carregado
    predicao = modelo_GBR_v2.predict(novos_dados_final)
    # Desfazer a padronização da variável alvo
    predicao_desnormalizada = scaler_y.inverse_transform(predicao.reshape(-1, 1))
    return predicao_desnormalizada[0][0]

# Configuração da interface do Streamlit
st.title("Previsão de Ganhos Mensais no YouTube")

nome_do_canal = st.text_input("Nome do Canal (opcional)")
total_de_inscritos = st.number_input("Total de Inscritos", min_value=1, max_value=289000000, step=1)
total_de_visualizacoes = st.number_input("Total de Visualizações", min_value=1, max_value=104005600000, step=1)
pais = st.selectbox("País", options=paises)
idade_do_canal = st.number_input("Idade do Canal (anos)", min_value=1, max_value=19, step=1)
total_de_visualizacoes_ultimos_30_dias = st.number_input("Visualizações Últimos 30 Dias", min_value=1, max_value=3592000000, step=1)
total_de_inscritos_ultimos_30_dias = st.number_input("Inscritos Últimos 30 Dias", min_value=1, max_value=27000000, step=1)

if st.button("Prever Ganhos Mensais"):
    # Preparar os dados para o modelo
    novos_dados_final = preparar_dados_para_modelo(total_de_inscritos, total_de_visualizacoes, pais, idade_do_canal,
                                                   total_de_visualizacoes_ultimos_30_dias, total_de_inscritos_ultimos_30_dias)
    # Prever os ganhos mensais estimados
    ganhos_mensais_estimados = prever_ganhos_mensais(novos_dados_final)
    # Ajustar valores negativos para zero
    if ganhos_mensais_estimados < 0:
        ganhos_mensais_estimados = 0
    # Exibição do resultado
    if nome_do_canal:
        st.markdown(f"## Os ganhos mensais estimados mínimos do canal '{nome_do_canal}' são: U$$ {ganhos_mensais_estimados:,.2f}")
    else:
        st.markdown(f"## Os ganhos mensais estimados mínimos para este canal são: U$$ {ganhos_mensais_estimados:,.2f}")
