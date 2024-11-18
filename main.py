#!/usr/bin/python3

from shutil import rmtree
from os import walk, mkdir
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
import json
#from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredMarkdownLoader, TextLoader
from models import Llama3, Qwen2
from chains import experimental_chain

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to input directory')
  flags.DEFINE_string('output_dir', default = 'processed', help = 'path to output directory')
  flags.DEFINE_enum('model', default = 'llama3', enum_values = {'llama3', 'qwen2'}, help = 'model to use')

def main(unused_argv):
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  tokenizer, llm = {
    'llama3': Llama3,
    'qwen2': Qwen2}[FLAGS.model](True)
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

