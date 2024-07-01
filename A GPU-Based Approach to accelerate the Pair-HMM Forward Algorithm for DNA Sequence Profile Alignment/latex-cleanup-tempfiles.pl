#!/usr/bin/perl -w
#  Author: Paul Horton
#  Created: 20210909
#  Updated: 20210909
#
#  Description:  Clean up some temporary latex files.
#
use feature qw(say);
use strict;
use warnings;


my $usage  =  "Usage: $0 [-n] latex-file...";


die $usage   if  grep {/^-h/} @ARGV;

my $dryRunP= shift    if @ARGV and $ARGV[0] eq '-n';

@ARGV  or  die $usage;


for my $filename (@ARGV){
    $filename=~ /[.]tex$/  or  die  "Expected tex filename but got '$filename'\n";
    
    my $filenameStem=  $filename=~ s/[.]tex$//r;

    for( qw(aux bbl bcf blg ilg log lof lot nav nlo nls out run.xml snm toc xdv) ){
        my $tempFilename= "$filenameStem.$_";
        if( -e $tempFilename ){
            say    "rm $tempFilename";
            system "rm $tempFilename"   unless $dryRunP;
        }
    }
}

