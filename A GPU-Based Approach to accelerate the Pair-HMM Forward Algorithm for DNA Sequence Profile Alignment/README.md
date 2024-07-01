# Template/Example for Master's or PhD Thesis

Developed by Paul Horton in 2021 for his lab in the computer science department at NCKU.  
Others are welcome to adapt this under the conditions of the Gnu Public License.

## Acknowledgements
The following two resources gave me ideas for this template.

*  https://github.com/wengan-li/ncku-thesis-template-latex
*  https://www.overleaf.com/latex/templates/imslab-thesis-template/rcfwchjpqbwx


# Requirements
* xelatex
* biber

# Recommended
* perl module Data::Lock  (to run the compile-thesis-from-scratch.pl script)
* emacs  (always recommended :)


# Usage

## Download & Maintain
Clone (not fork) this repository to your local computer.  
Place in your own gitlab/github repository with an informative
project name like: 'YOUR-NAME-master-thesis'.

## Copy template
    % cp thesis-example.tex thesis.tex
    % cp thesis-example.bib thesis.bib  #or copy in your own .bib file
    % sed s/-example//g compile-opts-example.tex > compile-opts.tex

## Edit Options
Select desired options by uncommenting lines in compile-opts.tex  

## Edit Documents
Edit thesis.tex as desired to change content.  
In particular change:
`\addbibresource{thesis-example.bib}` to point to your bib file.  
`\documentclass[PhD]{PHlab-thesis}` --> `\documentclass{PHlab-thesis}` unless you are a PhD student.  
`\documentclass{PHlab-thesis}` --> `\documentclass[BibLaTeXsortingNone]{PHlab-thesis}` if you do not want cited references to be sorted by author name.  
 
## Compile Document
    % perl compile-thesis-from-scratch.pl compile-opts.tex

or manually:

    % xelatex  -jobname thesis  compile-opts.tex
    % biber  thesis
    % xelatex  -jobname thesis  compile-opts.tex
    % xelatex  -jobname thesis  compile-opts.tex
    % makeindex  thesis.nlo  -s nomencl.ist  -o thesis.nls
    % xelatex  -jobname thesis  compile-opts.tex

Not all of that is usually necessary;__
if an edit does not affect citations, labels or nomenclature,  
just:

    % xelatex  -jobname thesis  compile-opts.tex

should be enough to update.


# File Summary

## Files Users Edit
    thesis.tex        Main thesis document.
    thesis.bib        To hold references
    config-opts.tex   Current compile options.  Better not to place it in git.

## Latex Formatting Files (users should not need to edit)
    PHlab-thesis.cls  class file.
    fonts-config.sty  configures fonts.
    frontmatter.tex   abstract, committee signature page, table of contents, nomenclature.

## Convenience Scripts
    compile-thesis-from-scratch.pl   Full compile from scratch.
    latex-cleanup-tempfiles.pl       Delete latex temp files.

## LaTex Tips
    PH-latex-tips.tex  A few tips on writing in LaTeX.

# Note on version control
Please do not place thesis.pdf or temporarily files like thesis.aux in git!
Version control is for source files.

