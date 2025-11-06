# Build a Multi-Agent Workshop

**[Home](Home.md)** - **[Prerequisites - Deployment and Setup >](./Module-00.md)**

## Learning Path

The workshop follows a progressive learning path with the following Modules

- [Module 0: Deployment and Project Setup](Module-00.md)
- [Module 1: Creating Your First Agent](Module-01.md)
- [Module 2: Agent Specialization](Module-02.md)
- [Module 3: Adding Memory to our Agents](Module-03.md)
- [Module 4: Making Memory Intelligent](Module-04.md)
- [Module 5: Observability & Experimentation](Module-05.md)
- [Module 6: Lessons Learned, Agent and Memory Future, Q&A](Module-06.md)

## Clean up

If done with workshop or sample application you can deprovision any Azure Services used.

Open a terminal and navigate to the /infra directory in this solution.

:bulb: The --force and --purge switches are required to delete the deployed Azure OpenAI models.

Type azd down.

```bash
azd down --force --purge
```