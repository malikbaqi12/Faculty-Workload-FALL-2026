# Faculty Workload Allocation and Course Recommendation System

Four files:

| File | Role |
|---|---|
| `faculty_workload_backend.ipynb` | Backend: data loading, Computing Core Course detection, faculty teaching history, content-based suitability scoring, constraint-based optimization, PDF report generation. Runnable in Google Colab for development/experimentation. |
| `app.py` | Frontend: single-file Streamlit app. Contains the same backend logic inlined so it runs standalone as a deployed app (no import dependency on the notebook). |
| `requirements.txt` | Python dependencies for both. |
| `README.md` | This file. |

## Why content-based + optimization, not a trained ML model

A single semester's workload sheet gives one labeled data point per
faculty member — not enough to train a classifier or ranking model that
generalizes to new courses. So the scoring is a transparent hybrid:

- **Content-based similarity** (TF-IDF + cosine similarity) between a
  faculty member's history of course names and each candidate course.
- **Rule-based adjustments** for the brief's constraints: a bonus if the
  course is a Computing Core Course, a penalty if the faculty member
  already taught that exact course last semester.
- **Constraint-based optimization** (integer programming via PuLP/CBC)
  turns the scores into a final, feasible allocation — enforcing the 9
  credit-hour cap, max 3 theory + 1 lab, max 1 repeated course, and max 2
  faculty per course. A scoring model alone can't guarantee these; a
  solver can.

As real multi-semester outcomes accumulate, the notebook's Section 8
sketches how to swap in a trained ranking model for the scoring step
without touching the optimization layer underneath it.

## Branding

`app.py` carries a navy blue and white theme (sidebar, buttons, tabs, and
headings are navy; the page background is white), the Iqra University
Chak Shahzad Campus logo displayed in the top-right of the header, and a
footer credit line: "Created by Abdul Baqi Malik (Lecturer, BSAI
Program), Department of Computing and Technology — Iqra University,
Chak Shahzad Campus, Islamabad." The logo is embedded in `app.py` as a
base64 string so the file stays self-contained — no separate image
asset to ship or lose track of. To swap the logo, replace the
`IU_LOGO_B64` constant near the top of Section 6 with a base64 encoding
of the new image (`base64.b64encode(open("logo.png","rb").read())`).

## Data formats

### Previous semester workload (`.xlsx` or `.docx`)

One row per faculty-course assignment. Column names are matched flexibly
(case/spacing-insensitive, common abbreviations), so a sheet or Word
table shaped like a typical department workload document loads as-is:

| Faculty Name | Course Code | Course Name | Theory/Lab | Semester | Cr.hr | Cn.hr |
|---|---|---|---|---|---|---|
| Dr. Tahir Ejaz | AIN476 | Artificial Intelligence for Robotics | Theory | 8th | 3 | 3 |

Handled automatically, for both formats:
- **Title rows above the real header** (e.g. a workload doc pasted from a
  department template, with "AI Programme Workload Spring 2026" and
  "Faculty Add" sitting above the actual column headers) — the loader
  scans the first several rows and auto-detects which one is the real
  header, instead of assuming row 0.
- **Merged/repeated faculty names** — forward-filled down the block.
- **Wrapped cells with embedded line breaks** (e.g. `"Ms. Samra\nJamil"`)
  — cleaned to a single space so the same person isn't split into two
  faculty records.
- **Section-divider rows inside a Word table** (e.g. a "CS Merge Labs" or
  "Combine" sub-heading row spanning the full table width) — detected
  (every cell in the row has identical text) and dropped, since they're
  not real faculty/course data.

### Scheme of Studies (`.xlsx`, `.docx`, or `.pdf`), one file per program — optional

| Course Code | Course Name | Credit Hours | Theory/Lab | Semester |
|---|---|---|---|---|
| AIN473 | Parallel and Distributed Computing | 3 | Theory | 6th |

PDFs are parsed with `pdfplumber`; if a scheme PDF has no machine-readable
tables (e.g. a scanned image), export it to Excel or Word instead.

**If you don't have a separate Scheme of Studies yet:** skip it. Both
`app.py` and the notebook fall back to using the previous semester's own
course list as the candidate pool for the upcoming semester, with a
clear warning that this fallback was used. The trade-off: it can only
recommend courses it has already seen taught before, not brand-new ones
— upload a real Scheme of Studies once you have it for a more accurate
result.

Sample template files aren't bundled in this 4-file drop — point the
notebook's `data/` paths or the app's file uploader at your own workload
and scheme files in these formats.

## Workload rules enforced

- Max 9 credit hours per faculty member.
- Max 3 theory courses + 1 lab course per faculty member.
- At most 1 previously-taught course repeated per faculty member.
- Every faculty member gets at least one Computing Core Course where
  feasible (soft-enforced — rewarded, not forced, so the solver stays
  feasible when core-course seats are scarce).
- A given course is recommended to at most 2 faculty members.
- Any Scheme of Studies course that can't be placed without violating a
  constraint is reported in "Unallocated Courses" with a reason, never
  silently dropped.

Computing Core Courses recognized out of the box: Programming
Fundamentals, Object-Oriented Programming, Data Structures, Computer
Networks, Database Management Systems, Analysis of Algorithms, Software
Engineering. Edit `COMPUTING_CORE_COURSES` (Section 1 of `app.py` /
Section 3 of the notebook) to add more.

## Running the app locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints, upload the workload file and one or
more scheme files in the sidebar, and click **Run allocation**.

## Deploying on Streamlit Community Cloud

1. Push these files to a GitHub repository.
2. On streamlit.io, "New app" → point it at the repo, branch, and `app.py`.
3. Streamlit Cloud installs `requirements.txt` automatically — PuLP's
   bundled CBC solver needs no separate system package.

## Using the backend notebook

Open `faculty_workload_backend.ipynb` in Google Colab (or Jupyter
locally). Section 6's upload cell lets you pick files in Colab — upload
the workload first (.xlsx or .docx), then any Scheme of Studies files
(optional); outside Colab it looks for sample files under `data/` next
to the notebook. Run all cells top to bottom — the later cells display
results and write (and, in Colab, download) a PDF report.

## Known limitations

- Suitability scoring is content-based (course-name similarity), not yet
  informed by faculty specialization tags, feedback scores, or research
  area — straightforward to fold into `_course_text` / `_faculty_profile_text`.
- The Computing Core Course requirement is soft; change `has_core[f] <=
  ...` to `has_core[f] == 1` in the optimization section if you'd rather
  it fail loudly when infeasible.
- PDF scheme extraction depends on the PDF having real text tables, not
  scanned images.
- The scheme-of-studies fallback (using the previous workload's own
  course list) can only recommend courses that were already taught
  before — it has no way to know about brand-new courses that aren't in
  any file you've uploaded.
