# Web and Files search

The globe and folder buttons under the message box let the model search the
web or folders you choose. Both are off until you click them on, and they
light up while on. Right-click the folder button to choose folders. Answers
show what was searched and cite their sources with numbers you can click.

Search results can be wrong, or written to mislead a model. They can't turn
on tools or reach other folders, but check the sources on anything that
matters.

## Web

DuckDuckGo works with no key. Brave Search and Tavily work with your own key,
which is kept in your Keychain and never shown to the model or saved in
chats.

What leaves your Mac: the search query, sent to the provider you picked, and
requests for the pages the model reads. Pages are limited to 2 MB and five
redirects, and local or private network addresses are refused.

On macOS 26 and newer with Apple's `container` tool installed, **Settings >
Search** can run web search and page reads in a sandbox, a small virtual
machine, so a page never reaches TUFF as anything but finished text. It is off
until you turn it on. See [Web sandbox](WEB_SANDBOX.md).

Once a chat includes your own files, images or folder results, web searches
can only use words you typed in your current message. That keeps private
material out of search queries.

DuckDuckGo sometimes blocks automated searches. TUFF tells you when that
happens instead of working around it. **Settings > Search** can pause web
search automatically while you're offline.

## Files

Files only reads folders you add. TUFF keeps a small text index in
`~/Library/Caches/TUFF/LocalSearch` (no embedding model) covering text,
Markdown, code, CSV, JSON, YAML and PDF. It skips hidden and build folders,
links, and big files, and pauses while the model is answering. Removing a
folder deletes its part of the index.

## Limits

Per answer: 4 rounds of tools, 3 calls per round, 8 web requests, 6 file
searches and 120 seconds of tool time. Stop cancels everything.

Search has been checked with real tool rounds on every model except GPT-OSS
120B and MiniMax M2.7. Those support tools but haven't been tried yet.
