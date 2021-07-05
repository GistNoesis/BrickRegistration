package LDraw::Parser;

use strict;
use warnings;

sub new {
    my ($class, $args) = @_;
    die "file required" unless $args->{file};
    return bless({
        file => $args->{file},
        ldraw_path => $args->{ldraw_path} // '/usr/share/ldraw',
        scale => $args->{scale} // 1,
        mm_per_ldu => $args->{mm_per_ldu} // 0.4,
        invert => $args->{invert} // 0,
        debug => $args->{debug} // 0,
        d_indent => $args->{d_indent} // 0,
        _invertnext => 0,
    }, $class);
}

sub _getter_setter {
    my ($self, $key, $value) = @_;
    if (defined $value) {
        $self->{$key} = $value;
    }
    return $self->{$key};
}

# The file to parse
sub file { return shift->_getter_setter('file', @_); }

# Where to find ldraw files
sub ldraw_path { return shift->_getter_setter('ldraw_path', @_); }

# Scale the model
sub scale { return shift->_getter_setter('scale', @_); }

# Number of mm per LDU (LDraw Unit)
sub mm_per_ldu { return shift->_getter_setter('mm_per_ldu', @_); }

# Invert this part
sub invert { return shift->_getter_setter('invert', @_); }

# Print debugging messages to stderr
sub debug { return shift->_getter_setter('debug', @_); }

# Indentation for debug messages (for subfiles)
sub d_indent { return shift->_getter_setter('d_indent', @_); }

use constant X => 0;
use constant Y => 1;
use constant Z => 2;

sub DEBUG {
    my ( $self, $message, @args) = @_;
    return if !$self->debug;
    my $indent = " " x $self->d_indent;
    if ( @args ) {
        $message = sprintf($message, @args);
    }
    print STDERR sprintf("%s%s\n", $indent, $message);
}

sub parse {
    my ( $self ) = @_;
    return $self->parse_file( $self->file );
}

sub parse_file {
    my ( $self, $file ) = @_;
    open( my $fh, '<', $file ) || die "$file: $!";
    $self->parse_handle( $fh );
    close $fh;
}

sub parse_handle {
    my ( $self, $handle ) = @_;
    while ( my $line = <$handle> ) {
        chomp $line;
        $self->parse_line( $line );
    }
}

sub parse_line {
    my ( $self, $line ) = @_;

    $line =~ s/^\s+//;

    if ( $line =~ /^([0-9]+)\s+(.+)$/ ) {
        my ( $line_type, $rest ) = ( $1, $2 );
        if ( $line_type == 0 ) {
            $self->parse_comment_or_meta( $rest );
        }
        elsif ( $line_type == 1 ) {
            $self->parse_sub_file_reference( $rest );
        }
        elsif ( $line_type == 2 ) {
            $self->parse_line_command( $rest );
        }
        elsif ( $line_type == 3 ) {
            $self->parse_triange_command( $rest );
        }
        elsif ( $line_type == 4 ) {
            $self->parse_quadrilateral_command( $rest );
        }
        elsif ( $line_type == 5 ) {
            $self->parse_optional( $rest );
        }
        else {
            warn "unhandled line type: $line_type";
        }
    }
}

sub parse_comment_or_meta {
    my ( $self, $rest ) = @_;
    my @items = split( /\s+/, $rest );
    my $first = shift @items;

    if ( $first && $first eq 'BFC' ) {
        $self->handle_bfc_command( @items );
    }
}

sub handle_bfc_command {
    my ( $self, @items ) = @_;

    my $first = shift @items;

    if (!$first) {
        $self->DEBUG('META: invalid BFC');
        return;
    }
    if ($first eq 'INVERTNEXT') {
        $self->{_invertnext} = 1;
        $self->DEBUG('META: INVERTNEXT found while invert[%d]', $self->invert);
        return;
    }
    if ($first eq 'CERTIFY') {
        if (!$items[0]) {
            $self->DEBUG('META: CERTIFY with no winding - default CCW');
            return;
        }
        #$self->DEBUG('META: BFC CERTIFY %s', $items[0]);
        return;
    }
    $self->DEBUG('META: Unknown BFC: %s', $items[0]);
}

sub parse_sub_file_reference {
    my ( $self, $rest ) = @_;
    # 16 0 -10 0 9 0 0 0 1 0 0 0 -9 2-4edge.dat
    my @items = split( /\s+/, $rest );
    my $color = shift @items;
    my $x = shift @items;
    my $y = shift @items;
    my $z = shift @items;
    my $a = shift @items;
    my $b = shift @items;
    my $c = shift @items;
    my $d = shift @items;
    my $e = shift @items;
    my $f = shift @items;
    my $g = shift @items;
    my $h = shift @items;
    my $i = shift @items;

#    / a d g 0 \   / a b c x \
#    | b e h 0 |   | d e f y |
#    | c f i 0 |   | g h i z |
#    \ x y z 1 /   \ 0 0 0 1 /

    my $mat = [
        $a, $b, $c, $x,
        $d, $e, $f, $y,
        $g, $h, $i, $z,
        0, 0, 0, 1,
    ];

    if ( scalar( @items ) != 1 ) {
        warn "um, filename is made up of multiple parts (or none)";
    }

    my $filename = lc( $items[0] );
    $filename =~ s/\\/\//g;

    my $p_filename = join( '/', $self->ldraw_path, 'p', $filename );
    my $hires_filename = join( '/', $self->ldraw_path, 'p/48', $filename );
    my $parts_filename = join( '/', $self->ldraw_path, 'parts', $filename );
    my $models_filename = join( '/', $self->ldraw_path, 'models', $filename );

    my $subpart_filename;
    if ( -e $hires_filename ) {
        $subpart_filename = $hires_filename;
    }
    elsif ( -e $p_filename ) {
        $subpart_filename = $p_filename;
    }
    elsif (-e $parts_filename ) {
        $subpart_filename = $parts_filename;
    }
    elsif ( -e $models_filename ) {
        $subpart_filename = $models_filename;
    }
    else {
        warn "unable to find file: $filename in normal paths";
        return;
    }

    my $det = mat4determinant($mat);
    my $invert = $self->invert;
    $self->DEBUG('FILE: %s BEFORE det[%d], invert[%d] _invertnext[%d]', $subpart_filename, $det, $invert, $self->{_invertnext});
    if ($det < 0) {
        $invert = 1;
    }
    if ($self->{_invertnext}) {
        $invert = $invert ? 0 : 1;
    }
    $self->DEBUG('FILE: %s AFTER  det[%d], invert[%d] _invertnext[%d]', $subpart_filename, $det, $invert, $self->{_invertnext});

    my $subparser = __PACKAGE__->new( {
        file       => $subpart_filename,
        ldraw_path => $self->ldraw_path,
        debug      => $self->debug,
        invert     => $invert,
        d_indent   => $self->d_indent + 2,
    } );
    $subparser->parse;
    $self->{_invertnext} = 0;

    for my $triangle ( @{ $subparser->{triangles} } ) {
        for my $vec ( @{ $triangle } ) {
            my @new_vec = mat4xv3( $mat, $vec );
            $vec->[0] = $new_vec[0];
            $vec->[1] = $new_vec[1];
            $vec->[2] = $new_vec[2];
        }
        push @{ $self->{triangles} }, $triangle;
    }
}

sub parse_line_command {
    my ( $self, $rest ) = @_;
}

sub parse_triange_command {
    my ( $self, $rest ) = @_;
    # 16 8.9 -10 58.73 6.36 -10 53.64 9 -10 55.5
    my @items = split( /\s+/, $rest );
    my $color = shift @items;
    if ($self->invert) {
        $self->_add_triangle([
            [$items[0], $items[1], $items[2]],
            [$items[6], $items[7], $items[8]],
            [$items[3], $items[4], $items[5]],
        ]);
    }
    else {
        $self->_add_triangle([
            [$items[0], $items[1], $items[2]],
            [$items[3], $items[4], $items[5]],
            [$items[6], $items[7], $items[8]],
        ]);
    }
}

sub parse_quadrilateral_command {
    my ( $self, $rest ) = @_;
    # 16 1.27 10 68.9 -6.363 10 66.363 10.6 10 79.2 7.1 10 73.27
    my @items = split( /\s+/, $rest );
    my $color = shift @items;
    my $x1 = shift @items;
    my $y1 = shift @items;
    my $z1 = shift @items;
    my $x2 = shift @items;
    my $y2 = shift @items;
    my $z2 = shift @items;
    my $x3 = shift @items;
    my $y3 = shift @items;
    my $z3 = shift @items;
    my $x4 = shift @items;
    my $y4 = shift @items;
    my $z4 = shift @items;
    if ($self->invert) {
        $self->_add_triangle([
            [$x1, $y1, $z1],
            [$x3, $y3, $z3],
            [$x2, $y2, $z2],
        ]);
        $self->_add_triangle([
            [$x3, $y3, $z3],
            [$x1, $y1, $z1],
            [$x4, $y4, $z4],
        ]);
    }
    else {
        $self->_add_triangle([
            [$x1, $y1, $z1],
            [$x2, $y2, $z2],
            [$x3, $y3, $z3],
        ]);
        $self->_add_triangle([
            [$x3, $y3, $z3],
            [$x4, $y4, $z4],
            [$x1, $y1, $z1],
        ]);
    }
}

sub _add_triangle {
    my ($self, $points) = @_;
    $points->[3] = $self->calc_surface_normal($points);
    push @{$self->{triangles}}, $points;
}

sub parse_optional {
    my ( $self, $rest ) = @_;
}

sub calc_surface_normal {
    my ($self, $points) = @_;
    my ($p1, $p2, $p3) = ($points->[0], $points->[1], $points->[2]);

    my ( $N, $U, $V ) = ( [], [], [] );

    $U->[X] = $p2->[X] - $p1->[X];
    $U->[Y] = $p2->[Y] - $p1->[Y];
    $U->[Z] = $p2->[Z] - $p1->[Z];

    $V->[X] = $p3->[X] - $p1->[X];
    $V->[Y] = $p3->[Y] - $p1->[Y];
    $V->[Z] = $p3->[Z] - $p1->[Z];

    $N->[X] = $U->[Y] * $V->[Z] - $U->[Z] * $V->[Y];
    $N->[Y] = $U->[Z] * $V->[X] - $U->[X] * $V->[Z];
    $N->[Z] = $U->[X] * $V->[Y] - $U->[Y] * $V->[X];

    return [$N->[X], $N->[Y], $N->[Z]];
}

sub mat4xv3 {
    my ( $mat, $vec ) = @_;

    my ( $a1, $a2, $a3, $a4,
         $b1, $b2, $b3, $b4,
         $c1, $c2, $c3, $c4 ) = @{ $mat };

    my ( $x_old, $y_old, $z_old ) = @{ $vec };

    my $x_new = $a1 * $x_old + $a2 * $y_old + $a3 * $z_old + $a4;
    my $y_new = $b1 * $x_old + $b2 * $y_old + $b3 * $z_old + $b4;
    my $z_new = $c1 * $x_old + $c2 * $y_old + $c3 * $z_old + $c4;

    return ( $x_new, $y_new, $z_new );
}

sub mat4determinant {
    my ($mat) = @_;
    my $a00 = $mat->[0];
    my $a01 = $mat->[1];
    my $a02 = $mat->[2];
    my $a03 = $mat->[3];
    my $a10 = $mat->[4];
    my $a11 = $mat->[5];
    my $a12 = $mat->[6];
    my $a13 = $mat->[7];
    my $a20 = $mat->[8];
    my $a21 = $mat->[9];
    my $a22 = $mat->[10];
    my $a23 = $mat->[11];
    my $a30 = $mat->[12];
    my $a31 = $mat->[13];
    my $a32 = $mat->[14];
    my $a33 = $mat->[15];
    my $b00 = $a00 * $a11 - $a01 * $a10;
    my $b01 = $a00 * $a12 - $a02 * $a10;
    my $b02 = $a00 * $a13 - $a03 * $a10;
    my $b03 = $a01 * $a12 - $a02 * $a11;
    my $b04 = $a01 * $a13 - $a03 * $a11;
    my $b05 = $a02 * $a13 - $a03 * $a12;
    my $b06 = $a20 * $a31 - $a21 * $a30;
    my $b07 = $a20 * $a32 - $a22 * $a30;
    my $b08 = $a20 * $a33 - $a23 * $a30;
    my $b09 = $a21 * $a32 - $a22 * $a31;
    my $b10 = $a21 * $a33 - $a23 * $a31;
    my $b11 = $a22 * $a33 - $a23 * $a32;

    return $b00 * $b11 - $b01 * $b10 + $b02 * $b09 + $b03 * $b08 - $b04 * $b07 + $b05 * $b06;
}

sub to_stl {
    my ( $self ) = @_;

    my $scale = $self->scale || 1;
    my $mm_per_ldu = $self->mm_per_ldu;

    my $stl = "";
    $stl .= "solid GiantLegoRocks\n";

    for my $triangle ( @{ $self->{triangles} } ) {
        my ( $p1, $p2, $p3, $n ) = @{ $triangle };
        $stl .= "facet normal " . join( ' ', map { sprintf( '%0.4f', $_ ) } @{ $n } ) . "\n";
        $stl .= "    outer loop\n";
        for my $vec ( ( $p1, $p2, $p3 ) ) {
            my @transvec = map { sprintf( '%0.4f', $_ ) } map { $_ * $mm_per_ldu * $scale } @{ $vec };
            $stl .= "        vertex " . join( ' ', @transvec ) . "\n";
        }
        $stl .= "    endloop\n";
        $stl .= "endfacet\n";
    }

    $stl .= "endsolid GiantLegoRocks\n";

    return $stl;
}

1;

__DATA__

## In handler for "!LDRAW":

    // If the scale of the object is negated then the triangle winding order
    // needs to be flipped.
    var matrix = currentParseScope.matrix;
    if (
        matrix.determinant() < 0 && (
            scope.separateObjects && isPrimitiveType( type ) ||
            ! scope.separateObjects
        ) ) {

        currentParseScope.inverted = ! currentParseScope.inverted;

    }

    triangles = currentParseScope.triangles;
    lineSegments = currentParseScope.lineSegments;
    conditionalSegments = currentParseScope.conditionalSegments;

    break;

## Handling sub-file:

    // Line type 1: Sub-object file
    case '1':

        var material = parseColourCode( lp );

        var posX = parseFloat( lp.getToken() );
        var posY = parseFloat( lp.getToken() );
        var posZ = parseFloat( lp.getToken() );
        var m0 = parseFloat( lp.getToken() );
        var m1 = parseFloat( lp.getToken() );
        var m2 = parseFloat( lp.getToken() );
        var m3 = parseFloat( lp.getToken() );
        var m4 = parseFloat( lp.getToken() );
        var m5 = parseFloat( lp.getToken() );
        var m6 = parseFloat( lp.getToken() );
        var m7 = parseFloat( lp.getToken() );
        var m8 = parseFloat( lp.getToken() );

        var matrix = new Matrix4().set(
            m0, m1, m2, posX,
            m3, m4, m5, posY,
            m6, m7, m8, posZ,
            0, 0, 0, 1
        );

        var fileName = lp.getRemainingString().trim().replace( /\\/g, "/" );

        if ( scope.fileMap[ fileName ] ) {

            // Found the subobject path in the preloaded file path map
            fileName = scope.fileMap[ fileName ];

        }    else {

            // Standardized subfolders
            if ( fileName.startsWith( 's/' ) ) {

                fileName = 'parts/' + fileName;

            } else if ( fileName.startsWith( '48/' ) ) {

                fileName = 'p/' + fileName;

            }

        }

        subobjects.push( {
            material: material,
            matrix: matrix,
            fileName: fileName,
            originalFileName: fileName,
            locationState: LDrawLoader.FILE_LOCATION_AS_IS,
            url: null,
            triedLowerCase: false,
            inverted: bfcInverted !== currentParseScope.inverted,
            startingConstructionStep: startingConstructionStep
        } );

        bfcInverted = false;

