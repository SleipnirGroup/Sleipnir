// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>

namespace slp {

/// Returns the number of columns a UTF-8 string occupies.
///
/// @param text UTF-8 string.
constexpr size_t display_width(std::string_view text) {
  // Count bytes that aren't continuation bytes, since each code point has
  // exactly one. We assume a code point is one column wide. See
  // https://www.rfc-editor.org/rfc/rfc3629#section-3.
  //
  // A continuation byte's top two bits are 0b10, so keep only those bits and
  // count the bytes where they're something else.
  return static_cast<size_t>(std::ranges::count_if(
      text, [](char c) { return (c & 0b11000000) != 0b10000000; }));
}

/// Word wraps text to the given width.
///
/// Paragraphs are separated by blank lines, and the line breaks within each
/// paragraph are replaced. Continuation lines are indented by the prefix's
/// width so they align with the text after the prefix, like a Google-style
/// docstring argument entry. Words longer than the width aren't split.
///
/// @param prefix String placed before the first line.
/// @param text UTF-8 text to wrap.
/// @param width Maximum line width in columns, including the indentation.
/// @return Wrapped text without a trailing newline.
constexpr std::string wrap(std::string_view prefix, std::string_view text,
                           size_t width) {
  constexpr std::string_view whitespace = " \n";
  const size_t indent = display_width(prefix);

  std::string wrapped{prefix};
  size_t column = indent;

  // Greedily fill lines. Append each word to the current line if it fits, or
  // start a new line indented to align with the text after the prefix.
  size_t end = 0;
  for (size_t begin = text.find_first_not_of(whitespace);
       begin != std::string_view::npos;
       begin = text.find_first_not_of(whitespace, end)) {
    // Find the whitespace since the previous word
    std::string_view gap = text.substr(end, begin - end);

    // Find the current word
    end = std::min(text.find_first_of(whitespace, begin), text.size());
    std::string_view word = text.substr(begin, end - begin);
    const size_t word_width = display_width(word);

    if (wrapped.size() == prefix.size()) {
      // Put the first word right after the prefix without a leading space
    } else if (gap.contains("\n\n")) {
      // Separate paragraphs with a blank line
      wrapped += "\n\n";
      wrapped.append(indent, ' ');
      column = indent;
    } else if (column + 1 + word_width > width) {
      // The space and word would overflow, so put the word on the next line
      wrapped += '\n';
      wrapped.append(indent, ' ');
      column = indent;
    } else {
      // The space and word fit, so put the word on the current line
      wrapped += ' ';
      ++column;
    }

    wrapped += word;
    column += word_width;
  }

  return wrapped;
}

}  // namespace slp
