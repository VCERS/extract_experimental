#!/usr/bin/python3

from shutil import rmtree
from os import walk, mkdir
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
import json
from transformers import AutoTokenizer
from langchain.document_loaders import UnstructuredMarkdownLoader, TextLoader
from models import TGI
from chains import experimental_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to input directory')
  flags.DEFINE_string('output_dir', default = 'processed', help = 'path to output directory')
  flags.DEFINE_string('host', default = 'http://localhost:8080/generate', help = 'url to TGI')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  llm = TGI(FLAGS.host)
  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
  exp_chain = experimental_chain(llm, tokenizer)
  for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
    for f in files:
      stem, ext = splitext(f)
      if ext not in ['.md', '.txt']: continue
      if ext == '.md':
        loader = UnstructuredMarkdownLoader(join(root, f), model = 'single', strategy = 'fast')
      elif ext == '.txt':
        loader = TextLoader(join(root, f))
      text = ' '.join([doc.page_content for doc in loader.load()])
      results = exp_chain.invoke({'text': text})
      with open(join(FLAGS.output_dir, stem + '.md'), 'w') as f:
        f.write(results)

if __name__ == "__main__":
  add_options()
  app.run(main)

