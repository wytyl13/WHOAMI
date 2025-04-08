



from whoami.tool.search.google_search_provider import GoogleSearch
from whoami.tool.base.base_tool import BaseTool


google_search = GoogleSearch(snippet_flag=0, search_config_path='/home/weiyutao/work/WHOAMI/whoami/scripts/test/search_config.yaml')


class WebSearchEngine(BaseTool):

    def _run(self, *args, **kwargs):

        query = kwargs.get("query")
        
        param = {
            "query": query
        }
        status, result = google_search(**param)

        return result



