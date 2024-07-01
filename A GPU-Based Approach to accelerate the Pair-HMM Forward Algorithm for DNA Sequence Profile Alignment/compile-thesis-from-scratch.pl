#!/usr/bin/perl -w
#  Copyright (C) 2021, Paul Horton
#  License: Gnu Public License
#  Created: 20210909
#  Updated: 20210910
#
#  Description:  Compile thesis latex file via a config latex file which includes it.
#
use feature qw(say);
use strict;
use warnings;
use Data::Lock 'dlock';


dlock my $usage  =  "Usage: $0 [-n] opt-config-file";


die $usage   if  grep {/^-h/} @ARGV;

my $dryRunP= shift    if @ARGV and $ARGV[0] eq '-n';

@ARGV  or  die $usage;


dlock my $optConfigFilename=  shift @ARGV;

$,= "\t";
die  "Unparsed command line args: '@ARGV'"   if @ARGV;




sub assert_existingLatexFile{
    for( $_[0] ){
        /[.]tex$/   or   die  "Expected filename ending in '.tex', but got '$_'";
        -e   or   die  "File '$_' not found";
    }
}


for( $optConfigFilename ){
    /[.]tex$/   or   die  "Expected filename ending in '.tex', but got '$_'";
    -e   or   die  "File '$_' not found";
}


open  my $optConfigFile,  '<',  $optConfigFilename
  or   die  "Failed to open file '$optConfigFilename'";


dlock my $mainSourceFilenameStem=
  do{
      my $inputFilename;
      my $sawOptCommitteeCommandP;
      for (<$optConfigFile>){
         $sawOptCommitteeCommandP= 't'  if /\\newcommand[*]?\\optCommittee/;
          s/%.*$//;
          if( /^  \\input\{  ( [-_.a-zA-Z]+ )  \}/x   ){
              die  "I expected $optConfigFilename to only \\input one file but found more.\n"   if $inputFilename;
              for ($inputFilename= $1){
                  s/[.]tex$//;
                  -e "$_.tex"   or   die  "File '$_' not found.";
              }
          }
      }
      $sawOptCommitteeCommandP   or  die  "\\newcommand*{optCommittee} not found in '$optConfigFilename'\n";
      $inputFilename;
};


sub run{
    say  "STAGE:  $_[0]";
    0 == system $_[0]   or   die  "Command failed: $_[0]"    unless $dryRunP;
}        
        


# Clean up temp files
for(  qw(aux bbl bcf blg ilg log lof lot nlo nls out run.xml toc xdv)){
    my $filename=  "$mainSourceFilenameStem.$_";
    if( -e $filename ){
        say  "rm $filename";
        system  "rm $filename"   unless $dryRunP;
    }
}


# Compile from scratch
run   "xelatex  -jobname $mainSourceFilenameStem  -halt-on-error  -no-pdf  $optConfigFilename";
run   "biber  $mainSourceFilenameStem";

run   "xelatex  -jobname $mainSourceFilenameStem  -halt-on-error  -no-pdf  $optConfigFilename";
run   "xelatex  -jobname $mainSourceFilenameStem  -halt-on-error  -no-pdf  $optConfigFilename";

run   "makeindex  $mainSourceFilenameStem.nlo  -s nomencl.ist  -o $mainSourceFilenameStem.nls";
run   "xelatex  -jobname $mainSourceFilenameStem  -halt-on-error  $optConfigFilename";

