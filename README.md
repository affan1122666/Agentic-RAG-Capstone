\# Agentic RAG Capstone



An Agentic Retrieval-Augmented Generation (RAG) assistant built with Python, LangGraph, local document retrieval, web search, LLM-based answer generation, critic verification, and an automatic revision loop.



\## Project Overview



This project implements an Agentic RAG system that can:



1\. Understand a user's research question.

2\. Create a research plan.

3\. Decide whether to use local documents, web sources, or both.

4\. Retrieve relevant information.

5\. Generate a draft answer.

6\. Verify the answer using a critic agent.

7\. Revise unsupported or incomplete answers.

8\. Re-check the revised answer.

9\. Produce a final verified answer.

10\. Save the complete agent execution trace to a log file.



The system is designed to reduce hallucinations by grounding answers in retrieved information and validating the generated response before finalization.



\## Architecture



```text

&#x20;                        USER QUESTION

&#x20;                             |

&#x20;                             v

&#x20;                        +----------+

&#x20;                        |  Planner |

&#x20;                        +----------+

&#x20;                             |

&#x20;                             v

&#x20;                     +----------------+

&#x20;                     | Source Router  |

&#x20;                     +----------------+

&#x20;                        /      |      \\

&#x20;                       /       |       \\

&#x20;                      v        v        v

&#x20;                 +--------+ +------+ +--------+

&#x20;                 | Local  | | Web  | |  Both  |

&#x20;                 |Retrieval| |Search| |Sources|

&#x20;                 +--------+ +------+ +--------+

&#x20;                      \\       |       /

&#x20;                       \\      |      /

&#x20;                        v     v     v

&#x20;                     +-------------+

&#x20;                     | Draft Answer|

&#x20;                     +-------------+

&#x20;                            |

&#x20;                            v

&#x20;                      +-----------+

&#x20;                      |  Critic   |

&#x20;                      +-----------+

&#x20;                         /     \\

&#x20;                      FAIL      PASS

&#x20;                       |         |

&#x20;                       v         v

&#x20;                  +----------+  +-----------+

&#x20;                  | Revision |  | Finalizer |

&#x20;                  +----------+  +-----------+

&#x20;                       |

&#x20;                       v

&#x20;                    Critic

&#x20;                       |

&#x20;                       v

&#x20;                    PASS

&#x20;                       |

&#x20;                       v

&#x20;                  Final Answer



&#x20;            Complete Trace --> logs/

