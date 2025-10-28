import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import joblib

# 🔢 Dados simulados
temperatura = [20, 22, 25, 27, 30, 32, 35, 37, 40]
vendas = [200, 240, 300, 330, 400, 450, 520, 580, 650]

# 📊 Criando o DataFrame
df = pd.DataFrame({'Temperatura': temperatura, 'Vendas': vendas})

# 🧠 Treinando o modelo
modelo = LinearRegression()
modelo.fit(df[['Temperatura']], df['Vendas'])

# 🔮 Fazendo uma previsão
nova_temp = 33
previsao = modelo.predict([[nova_temp]])
print(f"Previsão de vendas para {nova_temp}°C: {previsao[0]:.2f} unidades")

# 💾 Salvando o modelo
joblib.dump(modelo, 'modelo_regressao.joblib')

# 📈 Visualizando os dados
plt.scatter(df['Temperatura'], df['Vendas'], color='blue', label='Dados reais')
plt.plot(df['Temperatura'], modelo.predict(df[['Temperatura']]), color='red', label='Regressão Linear')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Sorvete')
plt.title('Relação entre Temperatura e Vendas')
plt.legend()
plt.savefig('grafico_regressao.png')  # Salva o gráfico para usar no README
plt.show()
