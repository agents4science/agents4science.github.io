"""LangChain agent wrapper for the scientific discovery pipeline."""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain.agents import create_tool_calling_agent, AgentExecutor


class LangAgent:
    """LangChain agent wrapper for scientific discovery workflows."""

    def __init__(
        self,
        name: str,
        role_description: str,
        tools: list[BaseTool] | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.name = name
        self.role_description = role_description
        self.tools = tools or []
        self.model = model

        self.llm = ChatOpenAI(model=self.model, temperature=temperature)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are the {self.name} agent. {self.role_description}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Build agent with tools if provided, otherwise use simple chain
        if self.tools:
            agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
            self._executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        else:
            self._executor = None

    def act(self, input_text: str) -> str:
        """Run the agent on the given input."""
        if self._executor is not None:
            result = self._executor.invoke({"input": input_text})
            return result["output"]

        # Simple chain for agents without tools
        chain = self.prompt | self.llm
        result = chain.invoke({"input": input_text, "agent_scratchpad": []})
        return result.content
