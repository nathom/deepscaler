"""
Tool utilities for DeepScaler's RL pipeline.

This module provides classes and functions to integrate tool use into the RL pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from deepscaler.rewards.wikisearch import search_wikipedia


@dataclass
class ToolCall:
    """Represents a model's request to use a tool."""

    tool_name: str
    arguments: Dict[str, Any]
    call_id: str


@dataclass
class ToolResponse:
    """Represents the response from a tool execution."""

    tool_name: str
    call_id: str
    content: str
    success: bool
    error_message: Optional[str] = None


class Tool(ABC):
    """Base class for all tools that can be called by the model."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        """Execute the tool with the given arguments and return the result."""
        pass

    def to_prompt_format(self) -> str:
        """Return a string representation of this tool for inclusion in prompts."""
        return f"Tool: {self.name}\nDescription: {self.description}\n"

    def to_usage_format(self) -> str:
        """Return usage examples and format instructions for this tool."""
        return f"""When you need to use the {self.name} tool, use the following Python function-like format:

```
{self.name}(parameter1="value", parameter2="value")
```

The system will process your tool call and provide a response that you can use in your reasoning:

```
# Result from {self.name}:
Response content from the {self.name} tool
```
"""


class SearchTool(Tool):
    """Tool that searches for information in a knowledge base."""

    def __init__(self, knowledge_base: Dict[str, str] = None):
        super().__init__(
            name="search", description="Search for information on a given topic."
        )
        # Simple mock knowledge base for demonstration
        self.knowledge_base = knowledge_base or {}

    def to_usage_format(self) -> str:
        """Return search-specific usage examples and format instructions."""
        return """When you need to search for information, use the following Python function-like format:

```
search(query="your search query here")
```

The system will process your search and provide results that you can use in your reasoning:

```
# Result from search:
Search results for 'your search query here':
- Information related to your query
- Additional information if available
```

For example, if you want to find information about a mathematical concept:

```
search(query="definition of eigenvalues")
```
"""

    def execute(self, arguments: Dict[str, Any]) -> str:
        """Execute the search with the given query."""
        query = arguments.get("query", "")
        if not query:
            return "Error: No search query provided."

        # Simple mock implementation - in real usage, this would query a database or API
        if query in self.knowledge_base:
            return self.knowledge_base[query]
        else:
            # Return a mock response
            return f"Search results for '{query}':\n- No specific information found for this query."


class WikipediaTitleSearchTool(Tool):
    """Tool that searches Wikipedia article titles."""

    def __init__(self, index_dir: Optional[str] = None):
        super().__init__(
            name="search_wikipedia_titles",
            description="Search for Wikipedia articles by title. Returns the most relevant article titles.",
        )

    def to_usage_format(self) -> str:
        """Return usage examples and format instructions."""
        return """When you need to find Wikipedia articles by title, use:

```
search_wikipedia_titles(query="your search query", max_results=3)
```

The system will return matching article titles:

```
# Result from search_wikipedia_titles:
Matching Wikipedia articles for 'your search query':
1. Article Title 1
   URL: https://en.wikipedia.org/wiki/Article_Title_1
2. Article Title 2
   URL: https://en.wikipedia.org/wiki/Article_Title_2
```

For example, to find articles about a specific topic:

```
search_wikipedia_titles(query="quantum physics")
```
"""

    def execute(self, arguments: Dict[str, Any]) -> str:
        """Execute the Wikipedia title search."""
        query = arguments.get("query", "")
        if not query:
            return "Error: No search query provided."

        max_results = arguments.get("max_results", 3)
        try:
            max_results = int(max_results)
        except (ValueError, TypeError):
            max_results = 3

        try:
            # Call the search function
            results = search_wikipedia(query=query, limit=max_results)

            if not results:
                return f"No Wikipedia articles found for '{query}'."

            # Format the results focusing on titles
            output = [f"Matching Wikipedia articles for '{query}':"]
            for i, result in enumerate(results, 1):
                output.append(f"{i}. {result['title']}")
                output.append(f"   URL: {result['url']}")

            return "\n".join(output)

        except Exception as e:
            return f"Error searching Wikipedia titles: {str(e)}"


class WikipediaContentSearchTool(Tool):
    """Tool that searches Wikipedia article content."""

    def __init__(self, index_dir: Optional[str] = None):
        super().__init__(
            name="search_wikipedia_content",
            description="Search for information within Wikipedia article content. Returns relevant excerpts from articles.",
        )

    def to_usage_format(self) -> str:
        """Return usage examples and format instructions."""
        return """When you need to search within Wikipedia article content, use:

```
search_wikipedia_content(query="your search query", max_results=2)
```

The system will return content from matching articles:

```
# Result from search_wikipedia_content:
Wikipedia content for 'your search query':
1. From "Article Title 1":
   [First paragraph of content...]

2. From "Article Title 2":
   [First paragraph of content...]
```

For example, to find detailed information about a concept:

```
search_wikipedia_content(query="quantum physics")
```
"""

    def execute(self, arguments: Dict[str, Any]) -> str:
        """Execute the Wikipedia content search."""
        query = arguments.get("query", "")
        if not query:
            return "Error: No search query provided."

        max_results = arguments.get("max_results", 2)
        try:
            max_results = int(max_results)
        except (ValueError, TypeError):
            max_results = 2

        try:
            # Call the search function
            results = search_wikipedia(query=query, limit=max_results)

            if not results:
                return f"No Wikipedia content found for '{query}'."

            # Format the results with focus on content
            output = [f"Wikipedia content for '{query}':"]
            for i, result in enumerate(results, 1):
                output.append(f"{i}. From \"{result['title']}\":")

                # Get first few paragraphs of content (truncated)
                content = result["text_content"]

                # Extract first few paragraphs (up to 500 chars)
                paragraphs = content.split("\n\n")
                short_content = "\n\n".join(paragraphs[:2])
                if len(short_content) > 500:
                    short_content = short_content[:500] + "..."

                # Format with proper indentation
                formatted_content = "\n   ".join(short_content.split("\n"))
                output.append(f"   {formatted_content}")

                output.append(f"   URL: {result['url']}")
                output.append("")  # Add blank line between results

            return "\n".join(output)

        except Exception as e:
            return f"Error searching Wikipedia content: {str(e)}"


class WikipediaSectionSearchTool(Tool):
    """Tool that searches for specific sections in Wikipedia articles."""

    def __init__(self, index_dir: Optional[str] = None):
        super().__init__(
            name="search_wikipedia_sections",
            description="Find table of contents (sections) within Wikipedia articles. Useful for locating headers, topics, or chapter titles.",
        )

    def to_usage_format(self) -> str:
        """Return usage examples and format instructions."""
        return """When you need to find specific sections or table of contents in Wikipedia articles, use:

```
search_wikipedia_sections(query="topic", max_results=2)
```

The system will return table of contents from matching articles:

```
# Result from search_wikipedia_sections:
Wikipedia sections for 'topic':
1. From "Article Title 1":
   Table of Contents:
   - Section 1
   - Section 2
   - Section 3

2. From "Article Title 2":
   Table of Contents:
   - Introduction
   - History
   - Applications
```

For example, to find the structure of articles about a specific topic:

```
search_wikipedia_sections(query="quantum physics")
```
"""

    def execute(self, arguments: Dict[str, Any]) -> str:
        """Execute the Wikipedia section search."""
        query = arguments.get("query", "")
        if not query:
            return "Error: No search query provided."

        max_results = arguments.get("max_results", 2)
        try:
            max_results = int(max_results)
        except (ValueError, TypeError):
            max_results = 2

        try:
            # Call the search function
            results = search_wikipedia(query=query, limit=max_results)

            if not results:
                return f"No Wikipedia articles found for '{query}'."

            # Format the results with focus on table of contents
            output = [f"Wikipedia sections for '{query}':"]
            for i, result in enumerate(results, 1):
                output.append(f"{i}. From \"{result['title']}\":")
                output.append("   Table of Contents:")

                # Extract sections from TOC
                toc = result.get("table_of_contents", [])
                if toc:
                    for section in toc:
                        # Format section based on level (indentation)
                        level = int(section.get("level", 1))
                        title = section.get("title", "")
                        indent = "   " + "  " * (level - 1)
                        output.append(f"{indent}- {title}")
                else:
                    output.append("   No table of contents available for this article")

                output.append(f"   URL: {result['url']}")
                output.append("")  # Add blank line between results

            return "\n".join(output)

        except Exception as e:
            return f"Error searching Wikipedia sections: {str(e)}"


class ToolRegistry:
    """Registry for available tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self.tools[tool.name] = tool

    def execute_tool(self, tool_call: ToolCall) -> ToolResponse:
        """Execute a tool call and return the response."""
        tool_name = tool_call.tool_name

        if tool_name not in self.tools:
            return ToolResponse(
                tool_name=tool_name,
                call_id=tool_call.call_id,
                content="",
                success=False,
                error_message=f"Tool '{tool_name}' not found.",
            )

        try:
            tool = self.tools[tool_name]
            result = tool.execute(tool_call.arguments)

            return ToolResponse(
                tool_name=tool_name,
                call_id=tool_call.call_id,
                content=str(result),
                success=True,
            )
        except Exception as e:
            return ToolResponse(
                tool_name=tool_name,
                call_id=tool_call.call_id,
                content="",
                success=False,
                error_message=str(e),
            )

    def get_tool_descriptions(self) -> str:
        """Return a formatted string with all tool descriptions for prompts."""
        if not self.tools:
            return ""

        descriptions = ["Available tools:"]
        for tool in self.tools.values():
            descriptions.append(tool.to_prompt_format())

        return "\n".join(descriptions)

    def get_tool_usage_guide(self) -> str:
        """Return a formatted string with all tool usage instructions."""
        if not self.tools:
            return ""

        usage_guide = ["Tool Usage Guide:"]
        for tool in self.tools.values():
            usage_guide.append(tool.to_usage_format())

        return "\n".join(usage_guide)

    def get_system_prompt_with_tools(self, base_prompt: str) -> str:
        """Generate a system prompt that includes tool descriptions and usage."""
        tool_descriptions = self.get_tool_descriptions()
        tool_usage = self.get_tool_usage_guide()

        full_prompt = f"""{base_prompt}

{tool_descriptions}

{tool_usage}

You should reference the information provided in tool responses to help solve the problem. Let's think step by step and output the final answer within \\boxed{{}}."""

        return full_prompt


def parse_tool_calls(text: str) -> List[ToolCall]:
    """
    Parse tool calls from model output in Python function-like format.

    This function extracts tool calls from text in formats like:
    search(query="your search query")
    or
    tool_name(param1="value1", param2="value2")
    """
    tool_calls = []

    # Parser for Python function-like calls
    import re
    import uuid

    # Match function-like tool calls
    tool_pattern = r"(\w+)\s*\(\s*(.*?)\s*\)"

    matches = re.finditer(tool_pattern, text, re.DOTALL)
    for match in matches:
        tool_name, args_text = match.groups()

        # Skip if this looks like a regular Python function in the output,
        # not a tool call (heuristic: if preceded by "def " or followed by ":")
        pre_context = text[max(0, match.start() - 4) : match.start()]
        post_context = text[match.end() : min(len(text), match.end() + 1)]

        if "def " in pre_context or post_context == ":":
            continue

        # Parse arguments from text
        # This is a simple parser for key=value pairs
        arguments = {}

        # Find all key="value" or key='value' patterns
        arg_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'
        arg_matches = re.finditer(arg_pattern, args_text)

        for arg_match in arg_matches:
            key, value = arg_match.groups()
            arguments[key] = value

        # If no structured arguments were found, treat the entire args text as raw_text
        if not arguments and args_text.strip():
            arguments = {"raw_text": args_text.strip()}

        # Generate a unique ID for this tool call
        call_id = str(uuid.uuid4())

        tool_calls.append(
            ToolCall(tool_name=tool_name, call_id=call_id, arguments=arguments)
        )

    return tool_calls


def insert_tool_responses(
    original_text: str, tool_responses: List[ToolResponse]
) -> str:
    """
    Insert tool responses into the original text.

    This function appends tool responses after their corresponding function-like calls,
    in a format that resembles a code comment with results.
    """
    modified_text = original_text

    for response in tool_responses:
        import re

        # Find the entire tool call including trailing whitespace/newline
        if response.tool_name in modified_text:
            # Find all instances of the tool name being called as a function
            tool_pattern = rf"{response.tool_name}\s*\([^)]*\)"
            matches = list(re.finditer(tool_pattern, modified_text))

            if matches:
                # For simplicity, just append after the last occurrence
                # In a more sophisticated implementation, you'd match the specific call ID
                match = matches[-1]

                # Format the response as a code block
                if response.success:
                    result_text = f"\n```\n# Result from {response.tool_name}:\n{response.content}\n```\n"
                else:
                    result_text = f"\n```\n# Error from {response.tool_name}:\n{response.error_message or 'Unknown error'}\n```\n"

                # Find where to insert the response
                insert_pos = match.end()

                # Insert the response text after the tool call
                modified_text = (
                    modified_text[:insert_pos]
                    + result_text
                    + modified_text[insert_pos:]
                )

    return modified_text

