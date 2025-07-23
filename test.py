from tavily import TavilyClient

tavily_client = TavilyClient(api_key="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")
response = tavily_client.search("特朗普经济政策")

extract_response = tavily_client.extract([item["url"] for item in response["results"]])


print(extract_response)