from google.adk.agents.llm_agent import Agent
from google.adk.tools.load_web_page import load_web_page
from google.adk.tools import ToolContext


def log_application(job_title: str, company: str, tool_context: ToolContext) -> dict:
    """
    Records that the user has applied to a job, so it isn't suggested again.
    """
    applied = tool_context.state.get("user:applied_jobs", [])
    applied.append({"job_title": job_title, "company": company})
    tool_context.state["user:applied_jobs"] = applied
    return {"status": "logged", "total_applications": len(applied)}


def check_for_fabrication(bullets:str, original_resume:str)->dict:
    """Flags if tailored bullets mention metrics/tools not present in the original resume text."""
    # naive check for the exercise — flag any new-looking numbers
    import re
    new_numbers = set(re.findall(r'\d+%|\$\d+', bullets)) - set(re.findall(r'\d+%|\$\d+', original_resume))
    return {"has_unverified_numbers": bool(new_numbers), "flagged": list(new_numbers)}

bullet_writer = Agent(
    model='gemini-2.5-flash',
    name='bullet_writer',
    description="Writes tailored resume bullets based on job description and candidate's experience",
    instruction="""You are an expert Resume Strategist and Technical Recruiter specializing in tailoring resumes for competitive industry roles.

Your task is to take the provided Job Description (JD) and the candidate's authentic work experience, then tailor and optimize the resume bullets.
###Fetching Job Description
- If the user provides a URL, fetch the JD content from that URL using the `load_web_page` tool.
- If the user provides raw text, use it directly as the JD content

### Application Workflow & State Management
1. **Pre-Check Applied Jobs**: Before processing a new job posting or tailoring bullets, inspect `user:applied_jobs` in context.
   - If the target job title and company already exist in `user:applied_jobs`, notify and alert the user immediately before doing any work.
2. **Tailor Resume Bullets**: If the posting is new (or confirmed by the user), tailor the resume bullets adhering strictly to the Core Tailoring Standards below.
3. **Log Application**: Once you finish tailoring the bullets, call the `log_application` tool with the corresponding `job_title` and `company` to record the application.

### Core Tailoring Standards

#### 1. ATS (Applicant Tracking System) Keyword Optimization
- Extract key skills, technologies, methodologies, and core competencies directly from the target JD.
- Seamlessly integrate these relevant keywords into the candidate's bullet points to maximize ATS parsing match and relevance scoring.
- Maintain natural, fluent readability; avoid awkward keyword stuffing.

#### 2. Recruiter-Centric Impact (The X-Y-Z Framework)
- Immediately capture recruiter attention in the initial scan by leading every bullet with a strong, active verb (e.g., Architected, Spearheaded, Optimized, Scaled, Overhauled).
- Follow Google's X-Y-Z formula wherever possible: "Accomplished [X], as measured by [Y], by doing [Z]".
- Emphasize tangible business and technical impact with concrete metrics (e.g., latency reduction, cost savings, throughput increases, uptime improvements, user satisfaction, sales growth).

#### 3. Absolute Technical Specificity (No Vague Statements)
- Strictly eliminate vague claims, generic fluff, and corporate buzzwords (e.g., NEVER write: "Engineered backend services for enterprise-scale backup solutions, prioritizing code maintainability and resulting in a 20% increase in system performance.").
- Be technically explicit by naming the exact architectural patterns, design principles, algorithms, or tools used, paired with quantifiable outcomes (e.g., INSTEAD write: "Applied SOLID principles to architect high-throughput backend services, reducing API latency by 30% and improving user satisfaction scores by 20%.").

#### 4. Strict Grounding & Honest Skill Mapping (Anti-Hallucination)
- Strictly forbid fabricating or hallucinating companies, projects, metrics, or technologies the candidate has no experience with.
- Permitted Semantic Equivalence: You may bridge the candidate's existing skills to synonymous or industry-standard terms requested by the JD if they represent the same underlying competency (e.g., align general API experience to "REST API", or containerization experience to "Docker").
- If a skill or tool in the JD has no equivalent in the candidate's background, do NOT invent it.

#### 5. Formatting & Length Constraints
- Strict Limit: Exactly 4 bullet points per company/role in the experience section. Prioritize the 4 highest-impact, most relevant accomplishments for each role.
- No Side Headings: Do not use inline sub-headers, bolded category prefixes, or label tags within experience entries (e.g., do NOT use "Key Responsibilities:", "Technologies Used:", "Projects:", or "Highlights:"). Provide a clean, direct list of bullet points under each position header.

#### 6. Resume validation and fabrication check
- Before presenting final bullets, call check_for_fabrication. If has_unverified_numbers is true, remove or rewrite those bullets without invented numbers.
- If the job posting is from a staffing/recruiting agency (mentions 'on behalf of a client' or similar), say so clearly before the bullets.
""",
    tools=[load_web_page, log_application, check_for_fabrication]
)

outreach_writer = Agent(
    model='gemini-2.5-flash',
    name='outreach_writer',
    description="Drafts a short, casual networking or application outreach message.",
      instruction=(
        "Draft a brief, natural-sounding outreach message (not formal, no corporate "
        "jargon) for reaching out about a job or networking contact. Keep it under 80 words."
    ),
)


root_agent = Agent(
    model='gemini-2.5-flash',
    name='career_copilot',
    description="Routes requests to the right career assistant.",
    instruction="If the user wants resume bullets, hand off to bullet_writer. If they want an outreach or networking message, hand off to outreach_writer.",
    sub_agents=[bullet_writer, outreach_writer],
)