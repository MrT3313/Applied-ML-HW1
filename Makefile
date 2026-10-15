PY_BIN ?= /opt/homebrew/Caskroom/miniconda/base/envs/homl3/bin
QUESTIONS := Q1 Q2

.PHONY: generate-submission-pdfs install-pdf-deps

# Renders each saved notebook (outputs as they are, no re-run) to HTML, prints it to a paginated PDF with
# headless Chromium, and writes SUBMISSIONS_PDFs/<question>/<notebook>.pdf. submission_pdf.j2 wraps long lines.
generate-submission-pdfs:
	@for q in $(QUESTIONS); do \
		mkdir -p SUBMISSIONS_PDFs/$$q; \
		$(PY_BIN)/jupyter nbconvert --to webpdf --template-file submission_pdf.j2 \
			--output-dir SUBMISSIONS_PDFs/$$q $$q/*.ipynb || exit 1; \
	done

# One time setup: the Playwright library nbconvert's webpdf exporter needs, plus its Chromium build.
install-pdf-deps:
	$(PY_BIN)/pip install "nbconvert[webpdf]"
	$(PY_BIN)/playwright install chromium
