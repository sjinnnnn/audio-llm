# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenization classes for LLaMA."""

import os
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple,Union, Set, Collection

from transformers.convert_slow_tokenizer import import_protobuf
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

from pathlib import Path
import base64
import logging
import tiktoken
import unicodedata

import re
from audio import *
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, TruncationStrategy, \
    TextInput, TextInputPair, PreTokenizedInput, PreTokenizedInputPair, TensorType, EncodedInput, EncodedInputPair

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import TextInput

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

LANGUAGES = {
    "en": "english",
    "ko": "korean",
}
num_reserved_special_tokens = 256

# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on


def _list_find(
        input_list: List[Any],
        candidates: Tuple[Any],
        start: int = 0,
):
    for i in range(start, len(input_list)):
        if input_list[i] in candidates:
            return i
    return -1

def _replace_closed_tag(
        input_tokens: List[Any],
        start_tags: Union[Any, Tuple[Any]],
        end_tags: Union[Any, Tuple[Any]],
        inclusive_replace_func: callable,
        exclusive_replace_func: callable = lambda x: x,
        audio_info: Dict = None
):
    if isinstance(start_tags, (str, int)):
        start_tags = (start_tags,)
    if isinstance(end_tags, (str, int)):
        end_tags = (end_tags,)
    assert len(start_tags) == len(end_tags)

    output_tokens = []
    end = 0
    audio_idx = 0
    while True:
        start = _list_find(input_tokens, start_tags, end)
        if start == -1:
            break
        output_tokens.extend(exclusive_replace_func(input_tokens[end: start]))
        tag_idx = start_tags.index(input_tokens[start])
        end = _list_find(input_tokens, (end_tags[tag_idx],), start)
        if end == -1:
            raise ValueError("Unclosed audio token")
        output_tokens.extend(inclusive_replace_func(input_tokens[start: end + 1], audio_info, audio_idx))
        end += 1
        audio_idx += 1
    output_tokens.extend(exclusive_replace_func(input_tokens[end:]))

    return output_tokens

def load_bpe_file(model_path: Path) -> dict[bytes, int]:
    """
    Load BPE file directly and return mergeable ranks.

    Args:
        model_path (Path): Path to the BPE model file.

    Returns:
        dict[bytes, int]: Dictionary mapping byte sequences to their ranks.
    """
    log = logging.getLogger(__name__)

    mergeable_ranks = {}

    with open(model_path, encoding="utf-8") as f:
        content = f.read()

    for line in content.splitlines():
        if not line.strip():  # Skip empty lines
            continue
        try:
            token, rank = line.split()
            mergeable_ranks[base64.b64decode(token)] = int(rank)
        except Exception as e:
            log.warning(f"Failed to parse line '{line}': {e}")
            continue

    return mergeable_ranks

class LlamaTokenizer(PreTrainedTokenizer):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is
    no padding token in the original model.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        sp_model_kwargs (`Dict[str, Any]`, `Optional`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not the default system prompt for Llama should be used.
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to add spaces between special tokens.
        legacy (`bool`, *optional*):
            Whether or not the `legacy` behavior of the tokenizer should be used. Legacy is before the merge of #24622
            and #25224 which includes fixes to properly handle tokens that appear after special tokens.
            Make sure to also set `from_slow` to `True`.
            A simple example:

            - `legacy=True`:
            ```python
            >>> from transformers import LlamaTokenizerFast

            >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=True, from_slow=True)
            >>> tokenizer.encode("Hello <s>.") # 869 is '▁.'
            [1, 15043, 29871, 1, 869]
            ```
            - `legacy=False`:
            ```python
            >>> from transformers import LlamaTokenizerFast

            >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=False, from_slow=True)
            >>> tokenizer.encode("Hello <s>.")  # 29889 is '.'
            [1, 15043, 29871, 1, 29889]
            ```
            Checkout the [pull request](https://github.com/huggingface/transformers/pull/24565) for more details.
        add_prefix_space (`bool`, *optional*, defaults to `True`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. Again, this should be set with `from_slow=True` to make sure it's taken into account.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        errors='replace',
        audio_start_tag='<audio>',
        audio_end_tag='</audio>',
        **kwargs,
    ):  
        self.mergeable_ranks = load_bpe_file(vocab_file)

        super().__init__(**kwargs)

        self.audio_start_tag = audio_start_tag
        self.audio_end_tag = audio_end_tag
        self.audio_pad_tag = "[[[AUDIO:modality]]]"

        self.AUDIO_ST = [
            '[[[AUDIO:modality]]]',
            # Transcription Tag
            "<|startoftranscript|>",  # Transcription
            "<|startofanalysis|>",  # Analysis
            # Task Tag
            "<|translate|>",
            "<|transcribe|>",
            "<|caption|>",
            "<|keyword|>",
            # Language Tag
            "<|unknown|>",  # unknown language
            *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
            "<|ko_tr|>",  
            # Timestamps Tag
            "<|notimestamps|>",
            "<|sil|>",
            # Output Instruction
            "<|caption_audiocaps|>",  # Audiocaps caption style
            "<|caption_clotho|>",  # Clotho caption style
            "<|audioset_ontology|>",  # Audioset ontology style
            "<|caption_plain|>",  # plain caption
            "<|itn|>",  # inversed text normalized
            "<|wo_itn|>",  # without inversed text normalized
            "<|startofentityvalue|>",
            "<|endofentityvalue|>",
            "<|startofentitytype|>",
            "<|endofentitytype|>",
            "<|named_entity_recognition|>",  # named entity recognition task
            "<|audio_grounding|>",
            "<|startofword|>",
            "<|endofword|>",
            "<|delim|>",  # delimiter of timestamps pair in audio grounding
            "<|emotion_recognition|>",  # emotion recognition
            "<|music_description|>",  # music description
            "<|note_analysis|>",  # note analysis
            "<|pitch|>",  # note analysis: pitch
            "<|velocity|>",  # note analysis: velocity
            "<|instrument|>",  # note analysis:  instrument
            "<|speaker_meta|>",  # meta information of speaker
            "<|song_meta|>",  # meta information of song
            "<|question|>",  # AQA: question
            "<|answer|>",  # AQA: answer
            "<|choice|>",  # AQA: answer choice
            "<|scene|>",  # scene recognition
            "<|event|>",  # sound event
            "<|vocal_classification|>",  # vocal classification
            "<|speech_understanding|>",  # speech language understanding
            "<|scenario|>",  # speech language understanding: scenario
            "<|action|>",  # speech language understanding: action
            "<|entities|>",  # speech language understanding: entities
            "<|speech_edit|>",  # speech edit
            audio_start_tag,
            audio_end_tag
        ]

        self.errors = errors
        
        num_base_tokens = len(self.mergeable_ranks)
        
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",  # end of message
            "<|eot_id|>",  # end of turn
            "<|python_tag|>",
            "<|image|>",
        ]
        reserved_tokens = [
            f"<|reserved_special_token_{2 + i}|>" for i in range(num_reserved_special_tokens - len(special_tokens+self.AUDIO_ST))
        ]

        special_tokens = special_tokens + reserved_tokens + self.AUDIO_ST

        self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}

        self.audio_start_id = self.special_tokens[self.audio_start_tag]
        self.audio_end_id = self.special_tokens[self.audio_end_tag]
        self.audio_pad_id = self.special_tokens[self.audio_pad_tag]
        print(f"audio_start_id: {self.audio_start_id}, "
              f"audio_end_id: {self.audio_end_id}, "
              f"audio_pad_id: {self.audio_pad_id}.")

        tiktok_enc = tiktoken.Encoding(
            name="llama",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = num_base_tokens + len(special_tokens)
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.eot_id: int = self.special_tokens["<|eot_id|>"]
        self.eom_id: int = self.special_tokens["<|eom_id|>"]
        self.python_tag_id = self.special_tokens["<|python_tag|>"]
        self.pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]
        self.stop_tokens = [
            self.eos_id,
            self.special_tokens["<|eom_id|>"],
            self.special_tokens["<|eot_id|>"],
        ]

        self.decoder = {
            v: k for k, v in self.mergeable_ranks.items()
        }  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = tiktok_enc  # type: tiktoken.Encoding

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['tokenizer']
        return state

    def __setstate__(self, state):
        # tokenizer is not python native; don't pass it; rebuild it
        self.__dict__.update(state)
        tiktok_enc = tiktoken.Encoding(
            "llama",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = tiktok_enc

    def __len__(self) -> int:
        return self.tokenizer.n_vocab
    
    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.tokenizer.n_vocab

    def get_vocab(self):
        return self.mergeable_ranks

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    def tokenize(
            self,
            text: str,
            allowed_special: Union[Set, str] = "all",
            disallowed_special: Union[Collection, str] = (),
            audio_info: Dict = None,
            **kwargs,
    ) -> List[Union[bytes, str]]:
       
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(
                text, allowed_special=allowed_special, disallowed_special=disallowed_special
        ):
            tokens.append(self.decoder[t])

        def _encode_audio_placeholder(audio_tokens, audio_info, audio_idx):

            assert audio_tokens[0] == self.audio_start_tag and audio_tokens[-1] == self.audio_end_tag          
            audio_token_span = audio_info['audio_span_tokens'][audio_idx]
            return [self.audio_start_tag] + [self.audio_pad_tag] * (audio_token_span - 2) + [self.audio_end_tag]

        return _replace_closed_tag(
            tokens,
            self.audio_start_tag,
            self.audio_end_tag,
            _encode_audio_placeholder,
            audio_info=audio_info
        )

   
    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[TextInput],
                List[TextInputPair],
                List[PreTokenizedInput],
                List[PreTokenizedInputPair],
                List[EncodedInput],
                List[EncodedInputPair],
            ],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_split_into_words: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs,
    ) -> BatchEncoding:

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        input_ids = []
        audio_info = kwargs.pop("audio_info", None)
        for pair_id in range(len(batch_text_or_text_pairs)):
            kwargs['audio_info'] = audio_info[pair_id]
            ids_or_pair_ids = batch_text_or_text_pairs[pair_id]
            # for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)




    def _tokenize(self, text: str, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        file_path = os.path.join(save_directory, "llama.tiktoken")
        with open(file_path, "w", encoding="utf8") as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + " " + str(v) + "\n"
                w.write(line)
        return (file_path,)
    
    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")
    
    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    @property
    def default_chat_template(self):
        """
        LLaMA uses [INST] and [/INST] to indicate user messages, and <<SYS>> and <</SYS>> to indicate system messages.
        Assistant messages do not have special tokens, because LLaMA chat models are generally trained with strict
        user/assistant/user/assistant message ordering, and so assistant messages can be identified from the ordering
        rather than needing special tokens. The system message is partly 'embedded' in the first user message, which
        results in an unusual token ordering when it is present. This template should definitely be changed if you wish
        to fine-tune a model with more flexible role ordering!

        The output should look something like:

        <bos>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST] Answer <eos><bos>[INST] Prompt [/INST] Answer <eos>
        <bos>[INST] Prompt [/INST]

        The reference for this chat template is [this code
        snippet](https://github.com/facebookresearch/llama/blob/556949fdfb72da27c2f4a40b7f0e4cf0b8153a28/llama/generation.py#L320-L362)
        in the original repository.
        """
        template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
            "{% set system_message = messages[0]['content'] %}"
            "{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}"
            "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
            "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% for message in loop_messages %}"  # Loop over all non-system messages
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
            "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
            "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' '  + content.strip() + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )
        template = template.replace("USE_DEFAULT_PROMPT", "true" if self.use_default_system_prompt else "false")
        default_message = DEFAULT_SYSTEM_PROMPT.replace("\n", "\\n").replace("'", "\\'")
        template = template.replace("DEFAULT_SYSTEM_MESSAGE", default_message)

        return template
    

    def extract_audio_urls(self, text):
        pattern = rf"{self.audio_start_tag}(.*?){self.audio_end_tag}"
        return re.findall(pattern, text)

    def process_audio(self, text):
        #pattern = rf"{re.escape(self.audio_start_tag)}(.*?){re.escape(self.audio_end_tag)}"
        pattern = rf"{self.audio_start_tag}(.*?){self.audio_end_tag}"
        audio_paths = re.findall(pattern, text)

        if len(audio_paths) > 0:
            audios, audio_lens, audio_span_tokens = [], [], []
            for audio_path in audio_paths:
                audio = load_audio(audio_path)
                L = (audio.shape[0] if audio.shape[0] <= 480000 else 480000)  # max_length < 30s
                mel_len = L // 160
                audio = pad_or_trim(audio.flatten())
                mel = log_mel_spectrogram(audio)
                audio_len_after_cnn = get_T_after_cnn(mel_len)
                audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
                audio_len = [audio_len_after_cnn, audio_token_num]
                audios.append(mel)
                audio_lens.append(audio_len)
                audio_span_tokens.append(audio_token_num + 2)  # add audio bos eos
            input_audio_lengths = torch.IntTensor(audio_lens)
            input_audios = torch.stack(audios, dim=0)
            return {"input_audios": input_audios,
                    "input_audio_lengths": input_audio_lengths,
                    "audio_span_tokens": audio_span_tokens,
                    "audio_urls": audio_paths}
        else:
            return None
