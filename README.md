# SistemaMultiAgentes Crew

Bem vindo ao projeto SistemaMultiAgente base sobre uso de sistemas multi agentes baseados em LLM com a ferramenta crewAI.

### Exemplo

Neste exemplo você vai encontrar uma equipe de pesquisa esportiva, composta por um pesquisador e um analista esportivo.

![Exemplo sistema com equipe de IA como componente](https://github.com/lucas-althoff/sistema_multi_agentes/blob/main/crew_pesquisa_futebol.png)


### Instalando

`git clone https://github.com/lucas-althoff/sistema_multi_agentes.git`

- Boa prática: Crie um ambiente para instalar crewai sem gerar conflitos!

`pip install crewai`

### Personalizando

**Adicione sua `GEMINI_API_KEY` no arquivo `.env`**

- Modifique `src/sistema_multi_agentes/config/agents.yaml` para definir seus agentes
- Modifique `src/sistema_multi_agentes/config/tasks.yaml` para definir suas tarefas
- Modifique `src/sistema_multi_agentes/crew.py` para adicionar sua própria lógica, ferramentas e argumentos específicos
- Modifique `src/sistema_multi_agentes/main.py` para adicionar entradas personalizadas para seus agentes e tarefas

## Executando o Projeto

Para iniciar sua equipe de agentes de IA e começar a execução das tarefas, execute isto a partir da pasta raiz do seu projeto:

```bash
$ python src\sistema_multi_agentes\main.py
```

Este comando inicializa a Crew do sistema_multi_agentes, montando os agentes e atribuindo-lhes tarefas conforme definido em sua configuração.

Este exemplo, não modificado, criará um arquivo `results/report.md` com a saída de uma pesquisa sobre LLMs na pasta raiz.

## Entendendo sua Equipe (Crew)

A Crew do sistema_multi_agentes é composta por múltiplos agentes de IA, cada um com papéis, metas e ferramentas únicas. Esses agentes colaboram em uma série de tarefas, definidas em `config/tasks.yaml`, aproveitando suas habilidades coletivas para alcançar objetivos complexos. O arquivo `config/agents.yaml` descreve as capacidades e configurações de cada agente em sua equipe.

## Outros projetos

Para que você experimente a configuração de outras equipes de múltiplos agentes foram adicionados cenários na pasta `exemplos_crews`