#!/usr/bin/python3
# coding: utf8
import re
import os
from bs4 import BeautifulSoup


# PANE kirjak või mittekirjak label juba puhastatud failidele

def kirjuta_faili(jarjend, outputdir_path, input_file):
    output_path = os.path.join(outputdir_path, os.path.basename(input_file))
    _, failinimi = os.path.split(input_file)
    print(output_path)
    with open(output_path, 'w') as fod:
        fod.write('mittekirjak\t' + ' '.join(jarjend))

def loe_faili(filename, valjundkaust):
    with open(filename, 'r') as fid:
        data = fid.readlines()
    return kirjuta_faili(data, valjundkaust, filename)

def process_directory(input_dir, output_dir):
    # kui väljundkausta ei eksisteeri, siis tee see
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # otsime sisendkaustast üles kõik url-laiendiga failid
    failid = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
              f.endswith('.url')]
    for f in failid:
        loe_faili(f, output_dir)

def main():
    # directory peal
    # input_dir = 'kasitsi_kirjak_all'
    input_dir = 'kasitsi_uusmeedia_all' # muuda siis ka kirjuta_faili funktsioonis label
    outputdir_path = '%s_labeliga' % input_dir
    process_directory(input_dir, outputdir_path)

    # yhe urli peal
    # loe_faili('Ovr3.url', 'lambikas')


if __name__ == '__main__':
    main()