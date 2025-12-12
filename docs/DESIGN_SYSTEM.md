# Dashboard Design System

## Color Palette

The dashboard uses a consistent dark theme across all pages with the following color scheme:

| Element                       | Color                  | HEX         | Usage                       |
| ----------------------------- | ---------------------- | ----------- | ------------------------- |
| **Main Background**           | Dark Slate             | **#0F172A** | Main content area background |
| **Sidebar Background**        | Dark Blue Grey         | **#1E293B** | Sidebar navigation area |
| **Text Color**                | White                  | **#FFFFFF** | All text content |
| **Accent Color**              | Sky Blue               | **#38BDF8** | Headers, buttons, highlights |
| **Card/Container Background** | Slightly lighter slate | **#1A2238** | Cards, containers, info boxes |

## Design Principles

### Consistency
- All pages use the same color scheme
- Sidebar maintains consistent styling across navigation
- Main content area uses uniform background

### Readability
- High contrast between text and backgrounds
- Accent color used strategically for emphasis
- Card backgrounds provide visual separation

### Professional Appearance
- Dark theme reduces eye strain
- Clean, modern interface
- Cohesive visual language

## Implementation

### CSS Variables
The design system uses CSS custom properties (variables) for easy maintenance:

```css
:root {
    --main-bg: #0F172A;
    --sidebar-bg: #1E293B;
    --text-color: #FFFFFF;
    --accent-color: #38BDF8;
    --card-bg: #1A2238;
}
```

### Styled Elements

1. **Headers (h1, h2, h3)**
   - Color: Accent color (#38BDF8)
   - Border: Accent color underline for h1

2. **Buttons**
   - Background: Accent color (#38BDF8)
   - Text: Dark background (#0F172A)
   - Hover: Lighter accent (#0EA5E9)

3. **Cards/Containers**
   - Background: Card background (#1A2238)
   - Border: Accent color (#38BDF8)
   - Shadow: Subtle glow with accent color

4. **Tables/DataFrames**
   - Background: Card background (#1A2238)
   - Headers: Sidebar background (#1E293B)
   - Text: White (#FFFFFF)

5. **Input Fields**
   - Background: Card background (#1A2238)
   - Border: Accent color (#38BDF8)
   - Text: White (#FFFFFF)

## Chart Styling

All matplotlib and seaborn charts are styled to match the dark theme:

- Figure background: Main background (#0F172A)
- Axes background: Card background (#1A2238)
- Text color: White (#FFFFFF)
- Accent color: Sky Blue (#38BDF8) for lines, bars, highlights
- Grid: Subtle sidebar background color

## Best Practices

1. **Always use the color variables** - Don't hardcode colors
2. **Maintain contrast** - Ensure text is readable on backgrounds
3. **Use accent color sparingly** - For emphasis and important elements
4. **Consistent spacing** - Use consistent padding and margins
5. **Card-based layout** - Group related content in cards with card background

## Future Enhancements

- [ ] Add theme toggle (light/dark mode)
- [ ] Custom color picker for accent color
- [ ] Additional chart color schemes
- [ ] Animation and transitions
- [ ] Responsive design improvements

